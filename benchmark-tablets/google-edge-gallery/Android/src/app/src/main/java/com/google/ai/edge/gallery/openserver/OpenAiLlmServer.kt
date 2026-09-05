/*
 * Copyright 2026 smolbenchmark (benchmark fork of Google AI Edge Gallery).
 *
 * Adds an OpenAI-compatible HTTP server (GET /v1/models, POST /v1/chat/completions with
 * JSON + SSE streaming, GET /healthz) that runs inference against LiteRT-LM models already
 * downloaded by the app, via the app's own LlmChatModelHelper. This mirrors Google's
 * first-party OpenAI server in the LiteRT-LM CLI (litert_lm serve / openai_handler.py)
 * but inside the Android app, so host tools like aiperf can benchmark on-device models
 * over `adb forward`.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.google.ai.edge.gallery.openserver

import android.content.Context
import android.util.Log
import com.google.ai.edge.gallery.common.getModelStorageDir
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.ModelAllowlist
import com.google.ai.edge.gallery.data.ModelDownloadStatus
import com.google.ai.edge.gallery.data.ModelDownloadStatusType
import com.google.ai.edge.gallery.data.RuntimeType
import com.google.ai.edge.gallery.data.resetInitialization
import com.google.ai.edge.gallery.runtime.ResultListener
import com.google.ai.edge.gallery.ui.llmchat.LlmChatModelHelper
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.gson.Gson
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject

/**
 * A minimal OpenAI-compatible server for benchmarking on-device LiteRT-LM models.
 *
 * Design notes:
 *  - It builds its own [Model] objects from the app's model allowlist (dev test file at
 *    /data/local/tmp/model_allowlist_test.json > cached allowlist on disk > embedded asset),
 *    so it is fully independent of the (activity-scoped) model-manager UI state.
 *  - Only models whose model file exists on disk are served (downloaded in the app's own UI).
 *  - Each request is stateless: the LiteRT conversation is reset with the request's message
 *    history as `initialMessages`, then a single turn is run for the last user message.
 *  - Requests for a given model are serialized (LiteRT-LM keeps ONE conversation per engine).
 *  - Token `usage` is estimated: completion tokens = number of streamed decode deltas (each
 *    LiteRT onMessage ≈ 1 token); prompt tokens = text-length heuristic (the 0.11.0 AAR
 *    exposes no tokenizer / token-count API).
 */
object OpenAiLlmServer {
  private const val TAG = "AGOpenAiServer"
  const val DEFAULT_PORT = 8080

  // Priority list of allowlist sources for building the model catalog.
  private const val TEST_ALLOWLIST_NAME = "model_allowlist_test.json" // /data/local/tmp
  private const val CACHED_ALLOWLIST_NAME = "model_allowlist.json" // modelsDir
  private const val EMBEDDED_ALLOWLIST_PATH = "allowlist/model_allowlist_1_0_19.json"

  private val gson = Gson()

  @Volatile private var serverSocket: ServerSocket? = null
  @Volatile private var acceptThread: Thread? = null
  @Volatile private var contextRef: Context? = null
  @Volatile private var catalog: List<Model> = emptyList()
  private val serving = ConcurrentHashMap<String, ServingModel>()
  private val requestCounter = AtomicLong(0)

  /** Observable status for the settings UI. */
  data class ServerStatus(
    val running: Boolean = false,
    val port: Int = -1,
    val modelCount: Int = 0,
    val downloadedModelCount: Int = 0,
    val error: String = "",
  )

  private val _status = MutableStateFlow(ServerStatus())
  val status: StateFlow<ServerStatus> = _status.asStateFlow()

  val isRunning: Boolean
    get() = serverSocket != null && acceptThread?.isAlive == true

  val boundPort: Int
    get() = serverSocket?.localPort ?: -1

  /** Per-model engine + serialization state. */
  private class ServingModel(val model: Model) {
    val lock = ReentrantLock()
    val inited = AtomicBoolean(false)
    @Volatile var initError: String? = null
    /** Effective accelerator the current engine was loaded with ("", "cpu", "gpu", ...). */
    @Volatile var activeAccelerator: String = ""
  }

  /** Starts the server asynchronously (safe to call from the main thread). */
  fun start(context: Context, port: Int = DEFAULT_PORT) {
    if (_status.value.running) return
    contextRef = context.applicationContext
    _status.value = ServerStatus(error = "Starting…")
    val worker =
      Thread {
        var ok = false
        try {
          val ctx = contextRef!!
          catalog = loadCatalog(ctx)
          Log.i(
            TAG,
            "Catalog loaded: ${catalog.size} llm_chat models: ${catalog.map { it.name }}",
          )
          // Try ports port..port+9 (8080 is often held by other dev servers on this tablet).
          var ss: ServerSocket? = null
          var chosenPort = -1
          for (p in port..port + 9) {
            try {
              ss = ServerSocket(p)
              chosenPort = p
              break
            } catch (e: Exception) {
              Log.w(TAG, "Port $p busy (${e.message}); trying next")
            }
          }
          if (ss == null) {
            _status.value = ServerStatus(error = "All ports $port..${port + 9} are busy")
            return@Thread
          }
          serverSocket = ss
          val t = Thread { acceptLoop(ss) }
          t.name = "openai-llm-server"
          t.isDaemon = true
          t.start()
          acceptThread = t
          _status.value =
            ServerStatus(
              running = true,
              port = chosenPort,
              modelCount = catalog.size,
              downloadedModelCount = downloadedModels().size,
            )
          Log.i(
            TAG,
            "OpenAI server listening on 0.0.0.0:$chosenPort (adb forward tcp:$chosenPort tcp:$chosenPort)",
          )
          ok = true
        } catch (e: Exception) {
          Log.e(TAG, "Server start failed", e)
          _status.value = ServerStatus(error = "Failed to start: ${e.message}")
        }
        if (!ok) {
          serverSocket = null
          acceptThread = null
        }
      }
    worker.name = "openai-server-starter"
    worker.start()
  }

  fun stop() {
    try {
      serverSocket?.close()
    } catch (_: IOException) {
    }
    serverSocket = null
    acceptThread = null
    _status.value =
      ServerStatus(
        modelCount = catalog.size,
        downloadedModelCount = if (contextRef != null) downloadedModels().size else 0,
      )
  }

  // ---------------------------------------------------------------------------------------------
  // Catalog
  // ---------------------------------------------------------------------------------------------

  private fun loadCatalog(context: Context): List<Model> {
    val raw = readAllowlistRaw(context)
    if (raw.isNullOrEmpty()) {
      Log.w(TAG, "No allowlist available; OpenAI server will report zero models")
      return emptyList()
    }
    return try {
      val allowlist = gson.fromJson(raw, ModelAllowlist::class.java)
      val models = mutableListOf<Model>()
      for (allowed in allowlist.models) {
        if (allowed.disabled == true) continue
        val taskTypes = allowed.taskTypes
        val isLlmChat = taskTypes.contains("llm_chat") || taskTypes.contains("llm_prompt_lab")
        val isLitert =
          allowed.runtimeType == null || allowed.runtimeType == RuntimeType.LITERT_LM
        if (!isLlmChat || !isLitert) continue
        val model = allowed.toModel()
        model.preProcess()
        models.add(model)
      }
      models
    } catch (e: Exception) {
      Log.e(TAG, "Failed to parse allowlist", e)
      emptyList()
    }
  }

  private fun readAllowlistRaw(context: Context): String? {
    // 1. Developer override pushed via adb.
    try {
      val devFile = File("/data/local/tmp", TEST_ALLOWLIST_NAME)
      if (devFile.exists()) return devFile.readText()
    } catch (e: Exception) {
      Log.w(TAG, "read dev allowlist failed", e)
    }
    // 2. Allowlist the app cached after a successful remote fetch.
    try {
      val cachedFile = File(getModelStorageDir(context), CACHED_ALLOWLIST_NAME)
      if (cachedFile.exists()) return cachedFile.readText()
    } catch (e: Exception) {
      Log.w(TAG, "read cached allowlist failed", e)
    }
    // 3. Embedded fallback bundled at build time.
    return try {
      context.assets.open(EMBEDDED_ALLOWLIST_PATH).bufferedReader().use { it.readText() }
    } catch (e: Exception) {
      Log.w(TAG, "read embedded allowlist failed", e)
      null
    }
  }

  private fun downloadStatusOf(model: Model): ModelDownloadStatus {
    return if (model.localFileRelativeDirPathOverride.isNotEmpty()) {
      ModelDownloadStatus(status = ModelDownloadStatusType.SUCCEEDED)
    } else {
      val file = File(model.getPath(contextRef!!))
      if (file.exists()) {
        ModelDownloadStatus(status = ModelDownloadStatusType.SUCCEEDED)
      } else {
        ModelDownloadStatus(status = ModelDownloadStatusType.NOT_DOWNLOADED)
      }
    }
  }

  private fun downloadedModels(): List<Model> {
    val ctx = contextRef ?: return emptyList()
    return catalog.filter {
      val f = File(it.getPath(ctx))
      f.exists() || it.localFileRelativeDirPathOverride.isNotEmpty()
    }
  }

  private fun resolveModel(name: String): ServingModel? {
    if (name.isBlank()) {
      val first = downloadedModels().firstOrNull() ?: return null
      return serving.computeIfAbsent(first.name) { ServingModel(first) }
    }
    val found = catalog.firstOrNull { it.name.equals(name, ignoreCase = true) }
      ?: return null
    return serving.computeIfAbsent(found.name) { ServingModel(found) }
  }

  // ---------------------------------------------------------------------------------------------
  // HTTP accept loop + request handling
  // ---------------------------------------------------------------------------------------------

  private fun acceptLoop(ss: ServerSocket) {
    while (!ss.isClosed) {
      val socket = try {
        ss.accept()
      } catch (e: Exception) {
        if (ss.isClosed) break
        continue
      }
      val t = Thread { handleConnection(socket) }
      t.name = "openai-conn"
      t.isDaemon = true
      t.start()
    }
  }

  private class HttpRequest(
    val method: String,
    val path: String,
    val headers: Map<String, String>,
    val body: String,
  )

  private class ApiError(val code: Int, message: String, val type: String = "invalid_request_error")
    : Exception(message)

  private fun handleConnection(socket: Socket) {
    try {
      socket.soTimeout = 0
      val input = BufferedInputStream(socket.getInputStream())
      val output = BufferedOutputStream(socket.getOutputStream())
      val request = try {
        readRequest(input)
      } catch (e: ApiError) {
        writeSimple(output, 400, "text/plain", e.message ?: "bad request")
        return
      }
      try {
        route(request, input, output)
      } catch (e: ApiError) {
        val err = JSONObject()
          .put(
            "error",
            JSONObject().put("message", e.message).put("type", e.type).put("code", JSONObject.NULL),
          )
        writeJson(output, e.code, err.toString())
      } catch (e: Exception) {
        Log.e(TAG, "Unhandled error handling ${request.path}", e)
        val err =
          JSONObject()
            .put("error", JSONObject().put("message", "Internal server error: ${e.message}").put("type", "server_error").put("code", JSONObject.NULL))
        writeJson(output, 500, err.toString())
      }
    } catch (e: Exception) {
      Log.d(TAG, "Connection error: ${e.message}")
    } finally {
      try {
        socket.close()
      } catch (_: IOException) {
      }
    }
  }

  private fun route(
    request: HttpRequest,
    input: InputStream,
    output: OutputStream,
  ) {
    val path = request.path
    val method = request.method
    when {
      method == "GET" && path == "/healthz" -> writeSimple(output, 200, "text/plain", "ok")
      method == "GET" && path == "/v1/models" -> handleModels(output)
      method == "POST" && path == "/v1/chat/completions" ->
        handleChatCompletions(request, input, output)
      else -> throw ApiError(404, "Not found: $method $path", "invalid_request_error")
    }
  }

  private fun handleModels(output: OutputStream) {
    val list = JSONArray()
    val now = System.currentTimeMillis() / 1000
    for (model in downloadedModels()) {
      list.put(
        JSONObject()
          .put("id", model.name)
          .put("object", "model")
          .put("created", now)
          .put("owned_by", "google")
          .put("permission", JSONArray()),
      )
    }
    val resp = JSONObject().put("object", "list").put("data", list)
    writeJson(output, 200, resp.toString())
  }

  // ---------------------------------------------------------------------------------------------
  // chat/completions
  // ---------------------------------------------------------------------------------------------

  private class TurnPlan(
    val systemInstruction: Contents?,
    val initialMessages: List<Message>,
    val userInput: String,
    val promptText: String,
  )

  private fun handleChatCompletions(
    request: HttpRequest,
    input: InputStream,
    output: OutputStream,
  ) {
    val body = JSONObject(request.body)
    val stream = body.optBoolean("stream", false)
    val requestedModel = body.optString("model", "")

    val sm = resolveModel(requestedModel)
      ?: throw ApiError(404, "model_not_found: unknown model '$requestedModel'. GET /v1/models to list available (downloaded) models. Available: ${catalog.joinToString { it.name }}")
    val ctx = contextRef!!
    if (downloadStatusOf(sm.model).status != ModelDownloadStatusType.SUCCEEDED) {
      throw ApiError(
        400,
        "Model '${sm.model.name}' is not downloaded on this device. Open the app -> AI Chat -> download '${sm.model.name}', then retry. (expected file: ${sm.model.getPath(ctx)})",
      )
    }

    val messagesArr = body.optJSONArray("messages")
      ?: throw ApiError(400, "missing required field: messages")
    val plan = buildTurnPlan(messagesArr)
    val temperature = if (body.has("temperature")) body.optDouble("temperature") else Double.NaN
    val acceleratorOverride = body.optString("accelerator", "")

    // Serialize per model: LiteRT keeps a single Conversation per engine.
    sm.lock.lock()
    try {
      ensureInitialized(ctx, sm, acceleratorOverride)
      applySamplingOverrides(sm.model, temperature)

      LlmChatModelHelper.resetConversation(
        model = sm.model,
        supportImage = false,
        supportAudio = false,
        systemInstruction = plan.systemInstruction,
        tools = emptyList(),
        enableConversationConstrainedDecoding = false,
        initialMessages = plan.initialMessages,
      )

      val id = "chatcmpl-" + requestCounter.incrementAndGet()
      val created = System.currentTimeMillis() / 1000
      val modelName = sm.model.name
      if (stream) {
        handleStreamingInference(output, sm, plan, id, created, modelName, body)
      } else {
        handleNonStreamingInference(output, sm, plan, id, created, modelName)
      }
    } finally {
      sm.lock.unlock()
    }
  }

  private fun buildTurnPlan(messagesArr: JSONArray): TurnPlan {
    if (messagesArr.length() == 0) throw ApiError(400, "messages must not be empty")
    val systemTexts = StringBuilder()
    val history = mutableListOf<Message>()
    val lastUser = AtomicReference<String?>()
    var lastRole = ""

    for (i in 0 until messagesArr.length()) {
      val m = messagesArr.getJSONObject(i)
      val role = m.optString("role", "")
      val text = extractText(m)
      if (text.isBlank()) continue
      when (role) {
        "system" -> {
          if (systemTexts.isNotEmpty()) systemTexts.append("\n")
          systemTexts.append(text)
        }
        "user" -> {
          history.add(Message.user(text))
          lastUser.set(text)
        }
        "assistant" -> {
          history.add(Message.model(text))
        }
        else -> {
          // tool / function roles ignored for this pure-chat server.
          Log.d(TAG, "Ignoring message with role '$role'")
        }
      }
      lastRole = role
    }

    if (lastRole != "user") {
      throw ApiError(400, "the last message must have role 'user' (model must have something to respond to)")
    }
    // The last user message is the turn input; everything before it is seeded history.
    val userInput = lastUser.get() ?: throw ApiError(400, "no user message found")
    history.removeAt(history.size - 1)

    val systemInstruction = if (systemTexts.isNotEmpty()) Contents.of(systemTexts.toString()) else null
    val promptText = buildString {
      if (systemTexts.isNotEmpty()) {
        append(systemTexts).append("\n")
      }
      for (h in history) {
        append(h.toString()).append("\n")
      }
      append(userInput)
    }
    return TurnPlan(systemInstruction, history, userInput, promptText)
  }

  private fun extractText(m: JSONObject): String {
    val content = m.opt("content") ?: return ""
    if (content is String) return content
    if (content is JSONArray) {
      val sb = StringBuilder()
      for (i in 0 until content.length()) {
        val part = content.optJSONObject(i) ?: continue
        if (part.optString("type") == "text") {
          sb.append(part.optString("text"))
        }
        // image_url / audio parts ignored in v1 (text-only server).
      }
      return sb.toString()
    }
    return ""
  }

  /**
   * Loads (or reloads) the engine for [sm], honouring a per-request `accelerator`
   * override. If the requested backend differs from the one the engine was loaded
   * with, the engine is torn down and reloaded — so a host can switch CPU/GPU
   * between benchmark runs (e.g. aiperf `--extra-inputs accelerator:cpu|gpu`)
   * without restarting the app.
   */
  private fun ensureInitialized(
    ctx: Context,
    sm: ServingModel,
    acceleratorOverride: String,
  ) {
    val desired = acceleratorOverride.ifBlank { null }
    val loaded = sm.model.instance != null
    val mismatch = loaded && desired != null && desired != sm.activeAccelerator

    if (loaded && !mismatch) {
      sm.inited.set(true)
      sm.initError = null
      return
    }

    if (loaded && mismatch) {
      Log.i(TAG, "Switching ${sm.model.name} accelerator ${sm.activeAccelerator} -> $desired")
      val cl = CountDownLatch(1)
      LlmChatModelHelper.cleanUp(sm.model) { cl.countDown() }
      cl.await(30, TimeUnit.SECONDS)
      sm.inited.set(false)
    }

    var lastError = ""
    // Try requested/default accelerator first, then CPU fallback.
    val attempts = if (desired != null) listOf(desired, "cpu") else listOf(null, "cpu")
    var ok = false
    for (accel in attempts) {
      val model = sm.model
      if (model.instance != null) {
        ok = true
        break
      }
      if (accel != null) {
        model.configValues = model.configValues.toMutableMap().apply {
          this[ConfigKeys.ACCELERATOR.label] = accel
        }
        Log.i(TAG, "Forcing accelerator=$accel for ${model.name}")
      }
      val latch = CountDownLatch(1)
      val errRef = AtomicReference<String?>(null)
      LlmChatModelHelper.initialize(
        context = ctx,
        model = model,
        taskId = "llm_chat",
        supportImage = false,
        supportAudio = false,
        onDone = { err ->
          if (err.isNotEmpty()) errRef.set(err)
          latch.countDown()
        },
        systemInstruction = Contents.of(""),
        tools = emptyList(),
        enableConversationConstrainedDecoding = false,
        coroutineScope = null,
      )
      latch.await(180, TimeUnit.SECONDS)
      if (model.instance != null) {
        Log.i(TAG, "Initialized ${model.name} with accelerator=${accel ?: "default"}")
        ok = true
        break
      }
      lastError = errRef.get() ?: "engine failed to initialize (instance null)"
      Log.w(TAG, "Init attempt accelerator=$accel failed: $lastError")
      if (sm.model.instance == null) {
        sm.model.resetInitialization()
      }
    }
    if (!ok) {
      sm.initError = lastError
      throw ApiError(500, "model load failed for ${sm.model.name}: $lastError", "server_error")
    }
    // Remember the effective accelerator so a later override can force a reload.
    sm.activeAccelerator =
      (sm.model.configValues[ConfigKeys.ACCELERATOR.label] as? String) ?: ""
    sm.inited.set(true)
    sm.initError = null
  }

  private fun applySamplingOverrides(model: Model, temperature: Double) {
    if (temperature.isNaN()) return
    val configs = model.configValues.toMutableMap()
    configs[ConfigKeys.TEMPERATURE.label] = temperature.coerceIn(0.0, 2.0).toString()
    model.configValues = configs
  }

  // ---------------------------------------------------------------------------------------------
  // Streaming
  // ---------------------------------------------------------------------------------------------

  private class SseResult(
    val output: OutputStream,
    val id: String,
    val created: Long,
    val modelName: String,
  )

  private fun handleStreamingInference(
    output: OutputStream,
    sm: ServingModel,
    plan: TurnPlan,
    id: String,
    created: Long,
    modelName: String,
    body: JSONObject,
  ) {
    writeSseHeaders(output)
    val full = StringBuilder()
    val deltaCount = AtomicInteger(0)
    val errorRef = AtomicReference<String?>(null)
    val latch = CountDownLatch(1)

    // Initial chunk carries the role so clients can open the message.
    writeSseData(
      output,
      chunkJson(id, created, modelName, roleChunkDelta("assistant"), finish = null),
    )

    val result: ResultListener = { partial, done, thinking ->
      try {
        if (partial.isNotEmpty()) {
          full.append(partial)
          deltaCount.incrementAndGet()
          writeSseData(output, chunkJson(id, created, modelName, contentChunkDelta(partial), finish = null))
        } else if (thinking != null && thinking.isNotEmpty()) {
          // thinking deltas are intentionally dropped from the OpenAI content stream
          full.append(thinking)
          deltaCount.incrementAndGet()
        }
        if (done) {
          latch.countDown()
        }
      } catch (e: Exception) {
        Log.w(TAG, "SSE write failed (client disconnected?): ${e.message}")
        LlmChatModelHelper.stopResponse(sm.model)
        errorRef.set("client_disconnected")
        latch.countDown()
      }
    }

    LlmChatModelHelper.runInference(
      model = sm.model,
      input = plan.userInput,
      resultListener = result,
      cleanUpListener = {},
      onError = { msg ->
        errorRef.set(msg)
        latch.countDown()
      },
      images = emptyList(),
      audioClips = emptyList(),
      coroutineScope = null,
      extraContext = null,
    )
    latch.await()

    val err = errorRef.get()
    if (err != null) {
      // We already committed to the SSE stream, so surface the error as a final data event.
      val errJson =
        JSONObject()
          .put(
            "error",
            JSONObject().put("message", err).put("type", "server_error").put("code", JSONObject.NULL),
          )
      writeSseData(output, errJson)
      writeSseDone(output)
      output.flush()
      return
    }

    val completionTokens = deltaCount.get()
    val promptTokens = estimateTokens(plan.promptText)
    val usage =
      JSONObject()
        .put("prompt_tokens", promptTokens)
        .put("completion_tokens", completionTokens)
        .put("total_tokens", promptTokens + completionTokens)
    // Final chunk: finish_reason.
    writeSseData(output, chunkJson(id, created, modelName, JSONObject(), finish = "stop"))
    // Usage chunk (benchmark clients read usage even without include_usage).
    val wantUsage =
      body.optJSONObject("stream_options")?.optBoolean("include_usage", false) == true ||
        body.optBoolean("_force_usage", false)
    if (wantUsage) {
      writeSseData(
        output,
        JSONObject()
          .put("id", id)
          .put("object", "chat.completion.chunk")
          .put("created", created)
          .put("model", modelName)
          .put("choices", JSONArray())
          .put("usage", usage),
      )
    }
    writeSseDone(output)
    output.flush()
    Log.i(TAG, "Streamed ${full.length} chars / ~$completionTokens decode deltas for $modelName")
  }

  private fun handleNonStreamingInference(
    output: OutputStream,
    sm: ServingModel,
    plan: TurnPlan,
    id: String,
    created: Long,
    modelName: String,
  ) {
    val full = StringBuilder()
    val deltaCount = AtomicInteger(0)
    val errorRef = AtomicReference<String?>(null)
    val latch = CountDownLatch(1)

    val result: ResultListener = { partial, done, thinking ->
      if (partial.isNotEmpty()) {
        full.append(partial)
        deltaCount.incrementAndGet()
      } else if (thinking != null && thinking.isNotEmpty()) {
        full.append(thinking)
        deltaCount.incrementAndGet()
      }
      if (done) latch.countDown()
    }

    LlmChatModelHelper.runInference(
      model = sm.model,
      input = plan.userInput,
      resultListener = result,
      cleanUpListener = {},
      onError = { msg ->
        errorRef.set(msg)
        latch.countDown()
      },
      images = emptyList(),
      audioClips = emptyList(),
      coroutineScope = null,
      extraContext = null,
    )
    latch.await()

    val err = errorRef.get()
    if (err != null) {
      val errJson =
        JSONObject()
          .put(
            "error",
            JSONObject().put("message", err).put("type", "server_error").put("code", null as String?),
          )
      writeJson(output, 500, errJson.toString())
      return
    }

    val completionTokens = deltaCount.get()
    val promptTokens = estimateTokens(plan.promptText)
    val usage =
      JSONObject()
        .put("prompt_tokens", promptTokens)
        .put("completion_tokens", completionTokens)
        .put("total_tokens", promptTokens + completionTokens)
    val resp =
      JSONObject()
        .put("id", id)
        .put("object", "chat.completion")
        .put("created", created)
        .put("model", modelName)
        .put(
          "choices",
          JSONArray().put(
            JSONObject()
              .put("index", 0)
              .put(
                "message",
                JSONObject().put("role", "assistant").put("content", full.toString()),
              )
              .put("finish_reason", "stop"),
          ),
        )
        .put("usage", usage)
    writeJson(output, 200, resp.toString())
    Log.i(TAG, "Non-stream reply ${full.length} chars / ~$completionTokens deltas for $modelName")
  }

  private fun chunkJson(
    id: String,
    created: Long,
    modelName: String,
    delta: JSONObject,
    finish: String?,
  ): JSONObject {
    val choice =
      JSONObject()
        .put("index", 0)
        .put("delta", delta)
        .put(
          "finish_reason",
          if (finish == null) JSONObject.NULL else finish,
        )
    return JSONObject()
      .put("id", id)
      .put("object", "chat.completion.chunk")
      .put("created", created)
      .put("model", modelName)
      .put("choices", JSONArray().put(choice))
  }

  private fun roleChunkDelta(role: String): JSONObject {
    return JSONObject().put("role", role)
  }

  private fun contentChunkDelta(content: String): JSONObject {
    return JSONObject().put("content", content)
  }

  // ---------------------------------------------------------------------------------------------
  // Token estimation (no tokenizer API in the 0.11.0 AAR)
  // ---------------------------------------------------------------------------------------------

  private fun estimateTokens(s: String): Int {
    if (s.isEmpty()) return 0
    var cjk = 0
    for (ch in s) {
      if (ch.code > 0x2e80) cjk++
    }
    val latin = s.length - cjk
    return kotlin.math.max(1, latin / 4 + cjk)
  }

  // ---------------------------------------------------------------------------------------------
  // HTTP primitives
  // ---------------------------------------------------------------------------------------------

  private fun readRequest(input: InputStream): HttpRequest {
    val reader = LineReader(input)
    val requestLine = reader.readHeaderLine()
    val parts = requestLine.split(" ")
    if (parts.size < 2) throw ApiError(400, "malformed request line")
    val method = parts[0]
    val path = parts[1]
    val headers = mutableMapOf<String, String>()
    while (true) {
      val line = reader.readHeaderLine()
      if (line.isEmpty()) break
      val idx = line.indexOf(':')
      if (idx > 0) {
        headers[line.substring(0, idx).trim().lowercase()] = line.substring(idx + 1).trim()
      }
    }
    var body = ""
    val cl = headers["content-length"]?.toIntOrNull() ?: 0
    if (cl > 0) {
      val bytes = ByteArray(cl)
      var read = 0
      while (read < cl) {
        val n = input.read(bytes, read, cl - read)
        if (n < 0) break
        read += n
      }
      body = String(bytes, 0, read, Charsets.UTF_8)
    }
    return HttpRequest(method, path, headers, body)
  }

  /** Reads CRLF (or bare LF) terminated lines for the HTTP header block. */
  private class LineReader(private val input: InputStream) {
    fun readHeaderLine(): String {
      val sb = StringBuilder()
      var prevCr = false
      while (true) {
        val b = input.read()
        if (b < 0) throw ApiError(400, "connection closed in headers")
        val c = b.toChar()
        if (c == '\n') {
          if (prevCr) sb.setLength(sb.length - 1) // drop \r
          return sb.toString()
        }
        sb.append(c)
        prevCr = c == '\r'
      }
    }
  }

  private fun writeSseHeaders(output: OutputStream) {
    val head =
      "HTTP/1.1 200 OK\r\n" +
        "Content-Type: text/event-stream\r\n" +
        "Cache-Control: no-cache\r\n" +
        "Connection: keep-alive\r\n" +
        "X-Accel-Buffering: no\r\n\r\n"
    output.write(head.toByteArray(Charsets.UTF_8))
    output.flush()
  }

  private fun writeSseData(output: OutputStream, json: JSONObject) {
    val payload = "data: " + json.toString() + "\n\n"
    output.write(payload.toByteArray(Charsets.UTF_8))
    output.flush()
  }

  private fun writeSseDone(output: OutputStream) {
    output.write("data: [DONE]\n\n".toByteArray(Charsets.UTF_8))
    output.flush()
  }

  private fun writeJson(output: OutputStream, code: Int, json: String) {
    val bytes = json.toByteArray(Charsets.UTF_8)
    val head =
      "HTTP/1.1 $code ${reason(code)}\r\n" +
        "Content-Type: application/json\r\n" +
        "Content-Length: ${bytes.size}\r\n" +
        "Connection: close\r\n\r\n"
    output.write(head.toByteArray(Charsets.UTF_8))
    output.write(bytes)
    output.flush()
  }

  private fun writeSimple(output: OutputStream, code: Int, type: String, text: String) {
    val bytes = text.toByteArray(Charsets.UTF_8)
    val head =
      "HTTP/1.1 $code ${reason(code)}\r\n" +
        "Content-Type: $type\r\n" +
        "Content-Length: ${bytes.size}\r\n" +
        "Connection: close\r\n\r\n"
    output.write(head.toByteArray(Charsets.UTF_8))
    output.write(bytes)
    output.flush()
  }

  private fun reason(code: Int): String =
    when (code) {
      200 -> "OK"
      400 -> "Bad Request"
      404 -> "Not Found"
      405 -> "Method Not Allowed"
      500 -> "Internal Server Error"
      else -> "OK"
    }
}
