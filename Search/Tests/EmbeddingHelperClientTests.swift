import XCTest
import Foundation
@testable import Search

/// `EmbeddingHelperClient` against a real child process: a script that speaks the helper's protocol (same command
/// line, same JSON lines) without a model, and misbehaves on demand.
///
/// The client always starts its helper as `<helper> --model <path> [--media <path>]`, so the script takes its
/// orders from the text in the model file: `ok`, `not-ready`, `die-on-request`, `hang-on-request` and so on (see
/// `fakeHelper` at the end of this file). It appends its process ID to a `pids` file beside the model, which is
/// how the tests know which processes were started and can check that none is left, alive or as a zombie.
///
/// A helper that dies used to take the test process (in production: Retrace) with it through SIGPIPE, so several
/// of these tests pass simply by getting to their next line.
final class EmbeddingHelperClientTests: XCTestCase {
    private var folder: URL!
    private var helperURL: URL!
    private var modelURL: URL!
    private var mediaURL: URL!
    private var clients: [EmbeddingHelperClient] = []

    override func setUp() async throws {
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/python3") else {
            throw XCTSkip("the stand-in helper needs /usr/bin/python3")
        }
        folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("retrace-embedding-helper-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        helperURL = folder.appendingPathComponent("FakeEmbeddingHelper")
        modelURL = folder.appendingPathComponent("model")
        mediaURL = folder.appendingPathComponent("media")
        try Self.fakeHelper.write(to: helperURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helperURL.path)
        try "image encoder".write(to: mediaURL, atomically: true, encoding: .utf8)
    }

    override func tearDown() async throws {
        for client in clients { client.shutdown() }
        clients.removeAll()
        if folder != nil {
            let left = await helpersLeft(of: (try? startedHelpers()) ?? [], within: .seconds(5))
            XCTAssertEqual(left, [], "helper processes were left behind")
            for pid in left { kill(pid, SIGKILL) }
            try? FileManager.default.removeItem(at: folder)
        }
    }

    // MARK: - Embedding

    func testTextsComeBackAsOneVectorEachInOrder() async throws {
        let client = try makeClient(mode: "ok")
        let nothing = try await client.embed([])
        XCTAssertEqual(nothing, [])
        XCTAssertEqual(try startedHelpers(), [], "nothing to embed shouldn't start a helper")

        let texts = ["hello world", "", "café ☕️ 日本語", String(repeating: "screen recording ", count: 400)]
        let vectors = try await client.embed(texts)

        XCTAssertEqual(vectors.count, 4)
        XCTAssertEqual(vectors.map(\.count), [8, 8, 8, 8])
        XCTAssertEqual(vectors, texts.map { expectedVector(for: $0) })

        // The same helper serves the next request.
        let more = try await client.embed(["second request"])
        XCTAssertEqual(more, [expectedVector(for: "second request")])
        XCTAssertEqual(try startedHelpers().count, 1)
    }

    /// A full text batch at a real model's size, both ways: the request and the reply are each several times
    /// what a pipe holds (64 KB).
    func testRequestAndReplyLargerThanAPipeBuffer() async throws {
        let client = try makeClient(mode: "wide")
        let texts = (0..<16).map { "frame \($0) " + String(repeating: "lorem ipsum ", count: 2_000 + $0) }

        let vectors = try await client.embed(texts)

        XCTAssertEqual(vectors.map(\.count), Array(repeating: 4096, count: 16))
        XCTAssertEqual(vectors, texts.map { expectedVector(for: $0, dimension: 4096) })
    }

    func testImagesNeedTheImageEncoder() async throws {
        let image = makeImage(width: 4, height: 3, seed: 7)

        let textOnly = try makeClient(mode: "ok")
        let error = await failure { try await textOnly.embed(images: [image]) }
        guard case .requestFailed = error else { return XCTFail("expected requestFailed, got \(String(describing: error))") }
        XCTAssertEqual(try startedHelpers(), [], "the helper shouldn't be started for a request it can't serve")
    }

    func testImagesComeBackAsOneVectorEachInOrder() async throws {
        let client = try makeClient(mode: "ok", media: true)
        let images = [makeImage(width: 4, height: 3, seed: 7), makeImage(width: 64, height: 40, seed: 200)]

        let vectors = try await client.embed(images: images)

        XCTAssertEqual(vectors.map(\.count), [8, 8])
        XCTAssertEqual(vectors, images.map(expectedVector))

        // Text goes through the same helper.
        let text = try await client.embed(["a caption"])
        XCTAssertEqual(text, [expectedVector(for: "a caption")])
        XCTAssertEqual(try startedHelpers().count, 1)
    }

    func testHelperErrorIsReportedAndTheNextCallWorks() async throws {
        let client = try makeClient(mode: "ok", media: true)
        let truncated = EmbeddingImage(width: 4, height: 3, rgb: Data(repeating: 1, count: 10))

        let error = await failure { try await client.embed(images: [truncated]) }
        guard case .requestFailed(let why) = error else {
            return XCTFail("expected requestFailed, got \(String(describing: error))")
        }
        XCTAssertTrue(why.contains("width x height x 3"), "the helper's own message should come through: \(why)")

        let image = makeImage(width: 4, height: 3, seed: 1)
        let vectors = try await client.embed(images: [image])
        XCTAssertEqual(vectors, [expectedVector(for: image)])
    }

    func testMissingHelperOrModelIsReportedWithoutStartingAnything() async throws {
        try "ok".write(to: modelURL, atomically: true, encoding: .utf8)

        let noHelper = EmbeddingHelperClient(
            helperURL: folder.appendingPathComponent("NoSuchHelper"),
            modelPath: modelURL.path
        )
        let helperError = await failure { try await noHelper.embed(["text"]) }
        guard case .helperNotFound = helperError else {
            return XCTFail("expected helperNotFound, got \(String(describing: helperError))")
        }

        let missingModel = folder.appendingPathComponent("no-such-model").path
        let noModel = EmbeddingHelperClient(helperURL: helperURL, modelPath: missingModel)
        let modelError = await failure { try await noModel.embed(["text"]) }
        guard case .modelNotFound(let path) = modelError else {
            return XCTFail("expected modelNotFound, got \(String(describing: modelError))")
        }
        XCTAssertEqual(path, missingModel)
        XCTAssertEqual(try startedHelpers(), [])
    }

    // MARK: - A helper that dies

    /// The SIGPIPE regression: the helper says it can't start and goes away. The client used to answer that by
    /// writing `quit` to it, and the write to the closed pipe killed the whole process.
    func testHelperThatCannotStartThrowsAndThisProcessSurvives() async throws {
        XCTAssertTrue(sigpipeIsDeadly(), "with SIGPIPE ignored this test proves nothing")
        let client = try makeClient(mode: "not-ready")

        let error = await failure { try await client.embed(["text"]) }

        guard case .failedToStart(let why) = error else {
            return XCTFail("expected failedToStart, got \(String(describing: error))")
        }
        XCTAssertEqual(why, "the model would not load")
        let started = try startedHelpers()
        XCTAssertEqual(started.count, 1)
        XCTAssertEqual(remaining(of: started), [], "the failed helper should be gone by the time the call throws")

        // Once the model is there, the same client starts a helper that works.
        try setMode("ok")
        let vectors = try await client.embed(["text"])
        XCTAssertEqual(vectors, [expectedVector(for: "text")])
    }

    /// The helper is gone before the request is written, so the request itself hits a closed pipe.
    func testHelperThatDiesBeforeTheRequestThrowsAndIsReplaced() async throws {
        XCTAssertTrue(sigpipeIsDeadly(), "with SIGPIPE ignored this test proves nothing")
        let client = try makeClient(mode: "die-after-ready")

        let error = await failure { try await client.embed(["text"]) }
        guard case .exited = error else { return XCTFail("expected exited, got \(String(describing: error))") }
        XCTAssertEqual(remaining(of: try startedHelpers()), [])

        let vectors = try await client.embed(["text"])
        XCTAssertEqual(vectors, [expectedVector(for: "text")])
        XCTAssertEqual(try startedHelpers().count, 2, "the second call should have started a new helper")
    }

    func testHelperThatDiesInTheMiddleOfARequestThrowsAndIsReplaced() async throws {
        XCTAssertTrue(sigpipeIsDeadly(), "with SIGPIPE ignored this test proves nothing")
        let client = try makeClient(mode: "die-on-request")

        let error = await failure { try await client.embed(["first", "second"]) }
        guard case .exited = error else { return XCTFail("expected exited, got \(String(describing: error))") }
        XCTAssertEqual(remaining(of: try startedHelpers()), [])

        let vectors = try await client.embed(["first", "second"])
        XCTAssertEqual(vectors, [expectedVector(for: "first"), expectedVector(for: "second")])
        XCTAssertEqual(try startedHelpers().count, 2, "the second call should have started a new helper")
    }

    func testHelperKilledWhileIdleIsReplacedOnTheNextCall() async throws {
        let client = try makeClient(mode: "ok")
        _ = try await client.embed(["warm up"])
        let first = try startedHelpers()
        XCTAssertEqual(first.count, 1)

        kill(first[0], SIGKILL)   // a crash, or the system reclaiming memory
        let left = await helpersLeft(of: first, within: .seconds(5))
        XCTAssertEqual(left, [], "a helper that died on its own should be reaped")

        let vectors = try await client.embed(["after the crash"])
        XCTAssertEqual(vectors, [expectedVector(for: "after the crash")])
        XCTAssertEqual(try startedHelpers().count, 2)
    }

    // MARK: - A helper that hangs

    func testHungRequestThrowsWithinTheRequestTimeout() async throws {
        let client = try makeClient(mode: "hang-on-request", requestTimeout: 1)
        // The helper is up after this, so what is timed below is the request alone, not the launch of Python.
        _ = try await client.embed(["warm up"])

        let clock = ContinuousClock()
        let start = clock.now
        let error = await failure { try await client.embed(["hang"]) }
        let waited = clock.now - start

        guard case .timedOut = error else { return XCTFail("expected timedOut, got \(String(describing: error))") }
        XCTAssertGreaterThanOrEqual(waited, .milliseconds(900), "it should have waited for the helper")
        XCTAssertLessThan(waited, .seconds(4))
        XCTAssertEqual(remaining(of: try startedHelpers()), [], "the hung helper should have been killed")

        try setMode("ok")
        let vectors = try await client.embed(["text"])
        XCTAssertEqual(vectors, [expectedVector(for: "text")])
    }

    func testARequestCancelledWhileWaitingItsTurnNeverReachesAHelper() async throws {
        let client = try makeClient(mode: "hang-on-request", requestTimeout: 1)
        _ = try await client.embed(["warm up"])

        // One request is with the helper, which will not answer. A second waits behind it and is cancelled
        // (the search it belonged to gave up).
        let hung = Task { try await client.embed(["hang"]) }
        try await Task.sleep(for: .milliseconds(100))
        let superseded = Task { try await client.embed(["no longer wanted"]) }
        try await Task.sleep(for: .milliseconds(50))
        superseded.cancel()

        do {
            _ = try await superseded.value
            XCTFail("a cancelled request should not produce vectors")
        } catch is CancellationError {
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
        _ = try? await hung.value   // times out; its helper is killed

        // Had the cancelled request been run after the hung one, it would have started a second helper.
        XCTAssertEqual(try startedHelpers().count, 1)
        XCTAssertEqual(remaining(of: try startedHelpers()), [])
    }

    /// The helper stops reading its stdin, so a request larger than the pipe can't even be written in full.
    func testHelperThatStopsReadingThrowsWithinTheRequestTimeout() async throws {
        let client = try makeClient(mode: "deaf", requestTimeout: 1)
        let large = String(repeating: "more than a pipe holds ", count: 20_000)
        _ = try await client.embed(["the one request it reads"])

        let clock = ContinuousClock()
        let start = clock.now
        let error = await failure { try await client.embed([large]) }
        let waited = clock.now - start

        guard case .timedOut = error else { return XCTFail("expected timedOut, got \(String(describing: error))") }
        XCTAssertGreaterThanOrEqual(waited, .milliseconds(900), "it should have waited for the helper")
        XCTAssertLessThan(waited, .seconds(4))
        XCTAssertEqual(remaining(of: try startedHelpers()), [], "the hung helper should have been killed")
    }

    func testHungStartThrowsWithinTheStartupTimeout() async throws {
        let client = try makeClient(mode: "hang-on-start", startupTimeout: 1)

        let clock = ContinuousClock()
        let start = clock.now
        let error = await failure { try await client.embed(["text"]) }
        let waited = clock.now - start

        guard case .timedOut = error else { return XCTFail("expected timedOut, got \(String(describing: error))") }
        XCTAssertGreaterThanOrEqual(waited, .milliseconds(900), "it should have waited for the helper")
        XCTAssertLessThan(waited, .seconds(4))
        XCTAssertEqual(remaining(of: try startedHelpers()), [], "the hung helper should have been killed")

        try setMode("ok")
        let vectors = try await client.embed(["text"])
        XCTAssertEqual(vectors, [expectedVector(for: "text")])
    }

    /// The helper prints a lot before it reads its request, while the request is more than its stdin pipe holds.
    /// Unless the client keeps reading while it writes, each side waits for the other until the request times out.
    func testHelperThatPrintsBeforeItReadsDoesNotStall() async throws {
        let client = try makeClient(mode: "chatty", requestTimeout: 5)
        let large = String(repeating: "more than a pipe holds ", count: 20_000)

        let vectors = try await client.embed([large, "small"])

        XCTAssertEqual(vectors, [expectedVector(for: large), expectedVector(for: "small")])
    }

    // MARK: - A helper that answers wrongly

    func testUnusableReplyThrowsAndTheNextCallStartsAFreshHelper() async throws {
        for mode in ["garbage-reply", "wrong-count", "wrong-id"] {
            let client = try makeClient(mode: mode)
            let before = try startedHelpers().count
            let texts = ["one", "two", "three"]

            let error = await failure { try await client.embed(texts) }
            guard case .requestFailed = error else {
                XCTFail("\(mode): expected requestFailed, got \(String(describing: error))")
                continue
            }
            // Only the helper that answered wrongly: the ones from earlier rounds are healthy and still up.
            let failed = Array(try startedHelpers().dropFirst(before))
            XCTAssertEqual(failed.count, 1, mode)
            XCTAssertEqual(remaining(of: failed), [], "\(mode): the helper should have been stopped")

            let vectors = try await client.embed(texts)
            XCTAssertEqual(vectors, texts.map { expectedVector(for: $0) }, mode)
            XCTAssertEqual(try startedHelpers().count, before + 2, "\(mode): the retry should use a new helper")
        }
    }

    // MARK: - Many callers

    func testConcurrentCallersEachGetTheirOwnAnswer() async throws {
        let client = try makeClient(mode: "ok", media: true)
        let callers = 24
        let texts: @Sendable (Int) -> [String] = { caller in
            ["caller \(caller) " + String(repeating: "a", count: caller), "caller \(caller), second text"]
        }
        let images = (0..<callers).map { makeImage(width: 2 + $0, height: 3, seed: UInt8($0)) }

        // Even callers embed text, odd ones a screenshot, all at once on the one client.
        let answers = try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for caller in 0..<callers {
                group.addTask {
                    if caller.isMultiple(of: 2) {
                        return (caller, try await client.embed(texts(caller)))
                    }
                    return (caller, try await client.embed(images: [images[caller]]))
                }
            }
            var answers: [Int: [[Float]]] = [:]
            for try await (caller, vectors) in group { answers[caller] = vectors }
            return answers
        }

        for caller in 0..<callers {
            let expected = caller.isMultiple(of: 2)
                ? texts(caller).map { expectedVector(for: $0) }
                : [expectedVector(for: images[caller])]
            XCTAssertEqual(answers[caller], expected, "caller \(caller) got someone else's answer")
        }
        XCTAssertEqual(try startedHelpers().count, 1, "all callers should share one helper")
    }

    // MARK: - Stopping

    func testIdleHelperIsStoppedAndTheNextCallStartsAnother() async throws {
        let client = try makeClient(mode: "ok", idleTimeout: 0.5)
        _ = try await client.embed(["first"])
        let first = try startedHelpers()
        XCTAssertEqual(first.count, 1)

        let left = await helpersLeft(of: first, within: .seconds(5))
        XCTAssertEqual(left, [], "the idle helper should have been stopped")

        let vectors = try await client.embed(["second"])
        XCTAssertEqual(vectors, [expectedVector(for: "second")])
        let both = try startedHelpers()
        XCTAssertEqual(both.count, 2)

        client.shutdown()
        let leftAfterShutdown = await helpersLeft(of: both, within: .seconds(5))
        XCTAssertEqual(leftAfterShutdown, [], "shutdown should stop the helper")
    }

    /// A helper that ignores `quit`, the end of its input and SIGTERM still has to go, and soon.
    func testShutdownStopsAHelperThatWillNotQuit() async throws {
        let client = try makeClient(mode: "stubborn")
        _ = try await client.embed(["text"])
        let started = try startedHelpers()

        client.shutdown()

        let left = await helpersLeft(of: started, within: .seconds(5))
        XCTAssertEqual(left, [], "shutdown should not leave a helper that ignores quit")

        let vectors = try await client.embed(["again"])
        XCTAssertEqual(vectors, [expectedVector(for: "again")])
    }

    func testReleasingTheClientStopsItsHelper() async throws {
        try setMode("ok")
        var client: EmbeddingHelperClient? = EmbeddingHelperClient(helperURL: helperURL, modelPath: modelURL.path)
        _ = try await client?.embed(["text"])
        let started = try startedHelpers()
        XCTAssertEqual(started.count, 1)

        client = nil

        let left = await helpersLeft(of: started, within: .seconds(5))
        XCTAssertEqual(left, [], "a helper shouldn't outlive its client")
    }

    // MARK: - Support

    /// A client for a helper in `mode`. Timeouts are generous unless the test is about one: the first launch of
    /// the system Python can be slow.
    private func makeClient(
        mode: String,
        media: Bool = false,
        idleTimeout: TimeInterval = 60,
        requestTimeout: TimeInterval = 20,
        startupTimeout: TimeInterval = 20
    ) throws -> EmbeddingHelperClient {
        try setMode(mode)
        let client = EmbeddingHelperClient(
            helperURL: helperURL,
            modelPath: modelURL.path,
            mediaModelPath: media ? mediaURL.path : nil,
            idleTimeout: idleTimeout,
            requestTimeout: requestTimeout,
            startupTimeout: startupTimeout
        )
        clients.append(client)
        return client
    }

    /// What the next helper to start will do. A helper reads this once, when it starts.
    private func setMode(_ mode: String) throws {
        try mode.write(to: modelURL, atomically: true, encoding: .utf8)
    }

    /// The process IDs of every helper started so far, in order.
    private func startedHelpers() throws -> [pid_t] {
        let file = folder.appendingPathComponent("pids")
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        return try String(contentsOf: file, encoding: .utf8).split(separator: "\n").compactMap { pid_t($0) }
    }

    /// Those of `pids` that still exist. A child that has exited but hasn't been reaped (a zombie) still does.
    private func remaining(of pids: [pid_t]) -> [pid_t] {
        pids.filter { kill($0, 0) == 0 || errno != ESRCH }
    }

    /// Waits up to `limit` for `pids` to be gone and returns the ones that aren't.
    private func helpersLeft(of pids: [pid_t], within limit: Duration) async -> [pid_t] {
        let clock = ContinuousClock()
        let deadline = clock.now + limit
        while !remaining(of: pids).isEmpty, clock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        return remaining(of: pids)
    }

    /// The `EmbeddingHelperError` that `call` throws. Anything else fails the test.
    private func failure(
        of call: () async throws -> [[Float]],
        file: StaticString = #filePath,
        line: UInt = #line
    ) async -> EmbeddingHelperError? {
        do {
            let vectors = try await call()
            XCTFail("expected an error, got \(vectors.count) vectors", file: file, line: line)
        } catch let error as EmbeddingHelperError {
            return error
        } catch {
            XCTFail("expected an EmbeddingHelperError, got \(error)", file: file, line: line)
        }
        return nil
    }

    /// Whether a write to a closed pipe would kill this process, as it would kill Retrace.
    private func sigpipeIsDeadly() -> Bool {
        var action = sigaction()
        sigaction(SIGPIPE, nil, &action)
        return unsafeBitCast(action.__sigaction_u.__sa_handler, to: Int.self) == 0   // SIG_DFL
    }

    private func makeImage(width: Int, height: Int, seed: UInt8) -> EmbeddingImage {
        let bytes = (0..<(width * height * 3)).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) }
        return EmbeddingImage(width: width, height: height, rgb: Data(bytes))
    }

    /// The vector the stand-in helper gives a text: computed here from the same rule, so a vector that was cut
    /// at the wrong place, decoded with the wrong byte order or handed to the wrong caller doesn't match.
    private func expectedVector(for text: String, dimension: Int = 8) -> [Float] {
        let bytes = Array(text.utf8)
        let sum = bytes.reduce(0) { $0 + Int($1) }
        return (0..<dimension).map { Float((sum + $0 * bytes.count) % 1000) }
    }

    /// The vector the stand-in helper gives a screenshot.
    private func expectedVector(for image: EmbeddingImage) -> [Float] {
        let sum = image.rgb.reduce(0) { $0 + Int($1) }
        return (0..<8).map { Float((image.width * 3 + image.height * 5 + sum + $0 * image.rgb.count) % 1000) }
    }

    /// The stand-in for `RetraceEmbeddingHelper`.
    private static let fakeHelper = #"""
    #!/usr/bin/python3
    # Stand-in for RetraceEmbeddingHelper: same command line, same JSON lines, no model.
    # The text in the file passed as --model says how to behave.
    import base64, json, os, signal, struct, sys, time

    arguments = sys.argv[1:]
    model = arguments[arguments.index("--model") + 1]
    media = arguments[arguments.index("--media") + 1] if "--media" in arguments else None
    folder = os.path.dirname(model)
    with open(model) as file:
        mode = file.read().strip()
    with open(os.path.join(folder, "pids"), "a") as file:
        file.write("%d\n" % os.getpid())

    dimension = 4096 if mode == "wide" else 8


    def emit(reply):
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


    def first_time():
        # A failure that should hit only the first helper leaves a marker for the ones after it.
        marker = os.path.join(folder, mode + ".done")
        if os.path.exists(marker):
            return False
        open(marker, "w").close()
        return True


    def die():
        # A process that dies closes its pipes a moment before its parent sees it as no longer running.
        # Holding that moment open makes every run go through it.
        os.close(0)
        os.close(1)
        time.sleep(0.3)
        os._exit(1)


    def hang():
        # Longer than any timeout under test, short enough not to linger if the test run itself is killed.
        time.sleep(30)
        os._exit(3)


    def text_vector(text):
        data = text.encode("utf-8")
        base = sum(data)
        return [float((base + j * len(data)) % 1000) for j in range(dimension)]


    def image_vector(image):
        rgb = base64.b64decode(image["rgb"])
        if len(rgb) != image["width"] * image["height"] * 3:
            raise ValueError("an image needs width, height and width x height x 3 bytes of rgb")
        base = image["width"] * 3 + image["height"] * 5 + sum(rgb)
        return [float((base + j * len(rgb)) % 1000) for j in range(dimension)]


    def encoded(vectors):
        floats = [value for vector in vectors for value in vector]
        return base64.b64encode(struct.pack("<%df" % len(floats), *floats)).decode("ascii")


    if mode == "hang-on-start":
        hang()
    if mode == "not-ready":
        emit({"ready": False, "error": "the model would not load"})
        die()
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    emit({"ready": True, "dim": dimension, "images": media is not None, "load_seconds": 0.0})
    if mode == "die-after-ready" and first_time():
        die()
    if mode == "chatty":
        sys.stdout.write("\n" * 300000)
        sys.stdout.flush()

    served = 0
    while True:
        if mode == "deaf" and served == 1:
            hang()
        line = sys.stdin.buffer.readline()
        if not line:
            if mode == "stubborn":
                hang()
            break
        request = json.loads(line)
        identifier, operation = request.get("id"), request.get("op")
        if operation == "quit":
            emit({"id": identifier, "ok": True})
            if mode == "stubborn":
                continue
            break
        if mode == "hang-on-request" and request.get("inputs") == ["hang"]:
            hang()
        if mode == "die-on-request" and first_time():
            die()
        if mode == "garbage-reply" and first_time():
            sys.stdout.write("this is not JSON\n")
            sys.stdout.flush()
            continue
        try:
            if operation == "embed":
                vectors = [text_vector(text) for text in request["inputs"]]
            elif operation == "image":
                if media is None:
                    raise ValueError("started without --media")
                vectors = [image_vector(image) for image in request["images"]]
            else:
                raise ValueError("unknown op %s" % operation)
        except ValueError as error:
            emit({"id": identifier, "ok": False, "error": str(error)})
            continue
        if mode == "wrong-count" and first_time():
            vectors = vectors[:-1]
        if mode == "wrong-id" and first_time():
            identifier += 1000
        emit({"id": identifier, "ok": True, "dim": dimension, "count": len(vectors), "vectors": encoded(vectors)})
        served += 1

    """#
}
