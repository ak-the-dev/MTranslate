import Foundation

struct BackendBridge {
    let repoRoot: URL

    private func appSupportURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["MTRANSLATE_APP_SUPPORT"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            if url.path.hasPrefix(repoRoot.path + "/") || url.path == repoRoot.path {
                return url
            }
        }
        return repoRoot.appendingPathComponent(".mtranslate_data")
    }

    private func defaultVLLMModelPath() -> String {
        appSupportURL()
            .appendingPathComponent("models")
            .appendingPathComponent("google_gemma_3_4b_it")
            .path
    }

    private func defaultInpaintModelPath() -> String {
        appSupportURL()
            .appendingPathComponent("models")
            .appendingPathComponent("sdxl_inpaint")
            .path
    }

    private func pythonCanImportVLLM(_ path: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["-c", "import vllm"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func detectVLLMPython() -> String? {
        let env = ProcessInfo.processInfo.environment
        var candidates: [String] = []
        if let p = env["MTRANSLATE_PYTHON"], !p.isEmpty { candidates.append(p) }
        candidates.append(repoRoot.appendingPathComponent(".venv_local/bin/python").path)
        candidates.append(repoRoot.appendingPathComponent(".venv/bin/python").path)
        candidates.append(URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".venv-vllm/bin/python").path)
        candidates.append("/usr/bin/python3")

        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path), pythonCanImportVLLM(path) {
                return path
            }
        }
        return nil
    }

    func initializeTranslationRuntime() throws -> String {
        guard let vllmPython = detectVLLMPython() else {
            throw NSError(domain: "MTranslateEditor", code: 21, userInfo: [
                NSLocalizedDescriptionKey: "No Python interpreter with `vllm` found. Set MTRANSLATE_PYTHON to an interpreter that can import vllm.",
            ])
        }

        let inpaintModel = defaultInpaintModelPath()
        guard FileManager.default.fileExists(atPath: inpaintModel) else {
            throw NSError(domain: "MTranslateEditor", code: 22, userInfo: [
                NSLocalizedDescriptionKey: "Missing SDXL inpaint model at \(inpaintModel). Pull it with: ./scripts/mtranslate.sh models pull-inpaint",
            ])
        }

        return "Runtime ready (Gemma3 via vLLM: \(vllmPython), SDXL: \(inpaintModel))."
    }

    private func pythonExecutable() -> String {
        let local = repoRoot.appendingPathComponent(".venv_local/bin/python").path
        if FileManager.default.isExecutableFile(atPath: local) {
            return local
        }
        let localVenv = repoRoot.appendingPathComponent(".venv/bin/python").path
        if FileManager.default.isExecutableFile(atPath: localVenv) {
            return localVenv
        }
        return "/usr/bin/python3"
    }

    private func cliLauncher() -> (executable: String, argsPrefix: [String]) {
        let script = repoRoot.appendingPathComponent("scripts/mtranslate.sh").path
        if FileManager.default.isExecutableFile(atPath: script) {
            return (script, [])
        }
        return (pythonExecutable(), ["-m", "mtranslate.cli"])
    }

    func manifestURL(jobID: String) -> URL {
        appSupportURL()
            .appendingPathComponent("jobs")
            .appendingPathComponent(jobID)
            .appendingPathComponent("manifest.json")
    }

    func loadManifest(jobID: String) throws -> JobManifest {
        let path = manifestURL(jobID: jobID)
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(JobManifest.self, from: data)
    }

    private func runCLI(_ args: [String]) throws -> String {
        let process = Process()
        process.currentDirectoryURL = repoRoot
        let launcher = cliLauncher()
        process.executableURL = URL(fileURLWithPath: launcher.executable)
        process.arguments = launcher.argsPrefix + args

        var env = ProcessInfo.processInfo.environment
        let rootPath = repoRoot.path
        if let existing = env["PYTHONPATH"], !existing.isEmpty {
            env["PYTHONPATH"] = "\(rootPath):\(existing)"
        } else {
            env["PYTHONPATH"] = rootPath
        }
        if env["MTRANSLATE_TRANSLATE_BACKEND"]?.isEmpty ?? true {
            env["MTRANSLATE_TRANSLATE_BACKEND"] = "vllm"
        }
        if env["MTRANSLATE_VLLM_MODEL"]?.isEmpty ?? true {
            let localModel = defaultVLLMModelPath()
            if FileManager.default.fileExists(atPath: localModel) {
                env["MTRANSLATE_VLLM_MODEL"] = localModel
            } else {
                env["MTRANSLATE_VLLM_MODEL"] = "google/gemma-3-4b-it"
            }
        }
        if env["MTRANSLATE_VLLM_ENABLE_REASONING"]?.isEmpty ?? true {
            env["MTRANSLATE_VLLM_ENABLE_REASONING"] = "1"
        }
        if env["MTRANSLATE_VLLM_REASONING_MODEL_HINT"]?.isEmpty ?? true {
            env["MTRANSLATE_VLLM_REASONING_MODEL_HINT"] = "gemma"
        }
        if env["MTRANSLATE_INPAINT_BACKEND"]?.isEmpty ?? true {
            env["MTRANSLATE_INPAINT_BACKEND"] = "diffusion"
        }
        if env["MTRANSLATE_INPAINT_MODEL"]?.isEmpty ?? true {
            env["MTRANSLATE_INPAINT_MODEL"] = defaultInpaintModelPath()
        }
        process.environment = env

        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err

        try process.run()
        process.waitUntilExit()

        let outData = out.fileHandleForReading.readDataToEndOfFile()
        let outText = String(data: outData, encoding: .utf8) ?? ""
        if process.terminationStatus != 0 {
            let errData = err.fileHandleForReading.readDataToEndOfFile()
            let stderrText = String(data: errData, encoding: .utf8) ?? ""
            let message = stderrText.isEmpty
                ? (outText.isEmpty ? "Unknown backend error" : outText)
                : stderrText
            throw NSError(domain: "MTranslateEditor", code: Int(process.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: message,
            ])
        }
        return outText
    }

    func applyEdits(jobID: String, payload: [String: Any]) throws {
        let dir = appSupportURL()
            .appendingPathComponent("ui_edits")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let file = dir.appendingPathComponent("\(jobID)_edits_\(Int(Date().timeIntervalSince1970)).json")
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: file)
        _ = try runCLI([
            "review",
            "--job", jobID,
            "--apply", file.path,
        ])
    }

    func runStage(jobID: String, pageID: String?, stage: String, allPages: Bool) throws {
        var args = [
            "step",
            "--job", jobID,
            "--stage", stage,
        ]
        if allPages {
            args.append("--all-pages")
        } else if let pageID {
            args += ["--page", pageID]
        }
        _ = try runCLI(args)
    }

    func createJob(inputDir: URL, outputDir: URL, export: String = "folder,pdf") throws -> String {
        let out = try runCLI([
            "create",
            "--input", inputDir.path,
            "--output", outputDir.path,
            "--export", export,
        ])
        let jobID = out
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .last(where: { !$0.isEmpty }) ?? ""
        if jobID.isEmpty {
            throw NSError(domain: "MTranslateEditor", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Backend returned an empty job id.",
            ])
        }
        return jobID
    }
}
