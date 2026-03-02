import Foundation
import AppKit
import CoreGraphics
import ImageIO

private struct ApprovalStore: Codable {
    var pages: [String: [String: Bool]]
}

@MainActor
final class EditorViewModel: ObservableObject {
    @Published var repoRoot: String
    @Published var jobID: String = ""
    @Published var pages: [PageManifest] = []
    @Published var selectedPageID: String = ""
    @Published var regions: [EditableRegion] = []
    @Published var blocks: [EditableBlock] = []
    @Published var selectedRegionID: String?
    @Published var selectedBlockID: String?
    @Published var stage: PreviewStage = .typeset
    @Published var overlayMode: OverlayMode = .regions
    @Published var selectedPipelineStage: PipelineStage = .ingest
    @Published var runStageForAllPages: Bool = false
    @Published var isDropTargeted: Bool = false
    @Published var statusText: String = ""
    @Published var isBusy: Bool = false

    private var manifest: JobManifest?
    private var approvalMap: [String: [String: Bool]] = [:]
    private var runtimeInitStarted = false

    init() {
        if let envRoot = ProcessInfo.processInfo.environment["MTRANSLATE_REPO_ROOT"], !envRoot.isEmpty {
            repoRoot = envRoot
            return
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        if let found = EditorViewModel.findRepoRoot(start: cwd) {
            repoRoot = found.path
        } else {
            repoRoot = FileManager.default.currentDirectoryPath
        }
    }

    private var bridge: BackendBridge {
        BackendBridge(repoRoot: URL(fileURLWithPath: repoRoot))
    }

    func initializeRuntimeOnLaunch() {
        guard !runtimeInitStarted else { return }
        runtimeInitStarted = true
        statusText = "Initializing translation runtime..."
        let bridge = self.bridge
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let message = try bridge.initializeTranslationRuntime()
                DispatchQueue.main.async {
                    self?.statusText = message
                }
            } catch {
                DispatchQueue.main.async {
                    self?.statusText = "Runtime init failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func approvalsURL() -> URL? {
        let id = jobID.trimmingCharacters(in: .whitespacesAndNewlines)
        if id.isEmpty { return nil }
        return URL(fileURLWithPath: repoRoot)
            .appendingPathComponent(".mtranslate_data")
            .appendingPathComponent("jobs")
            .appendingPathComponent(id)
            .appendingPathComponent("review")
            .appendingPathComponent("approvals.json")
    }

    private func loadApprovals() {
        guard let url = approvalsURL() else {
            approvalMap = [:]
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            approvalMap = [:]
            return
        }
        if let store = try? JSONDecoder().decode(ApprovalStore.self, from: data) {
            approvalMap = store.pages
        } else {
            approvalMap = [:]
        }
    }

    private func saveApprovals() {
        guard let url = approvalsURL() else { return }
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let store = ApprovalStore(pages: approvalMap)
            let data = try JSONEncoder().encode(store)
            try data.write(to: url)
        } catch {
            statusText = "Failed saving approvals: \(error.localizedDescription)"
        }
    }

    var selectedPage: PageManifest? {
        pages.first(where: { $0.page_id == selectedPageID })
    }

    var imageSize: CGSize {
        if let page = selectedPage, let w = page.width, let h = page.height {
            return CGSize(width: w, height: h)
        }
        if let path = currentImagePath(), let image = NSImage(contentsOfFile: path) {
            return image.size
        }
        return CGSize(width: 1000, height: 1400)
    }

    func loadJob() {
        guard !jobID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusText = "Enter a job id."
            return
        }
        isBusy = true
        defer { isBusy = false }

        do {
            let loaded = try bridge.loadManifest(jobID: jobID)
            manifest = loaded
            pages = loaded.pages.values.sorted(by: { $0.index < $1.index })
            loadApprovals()
            if selectedPageID.isEmpty || !pages.contains(where: { $0.page_id == selectedPageID }) {
                selectedPageID = pages.first?.page_id ?? ""
            }
            loadSelectedPageState()
            autoSelectStageForSelectedPage()
            statusText = "Loaded job \(jobID)."
        } catch {
            statusText = "Failed loading job: \(error.localizedDescription)"
        }
    }

    func reloadPage() {
        guard !jobID.isEmpty else { return }
        loadJob()
    }

    func importDroppedURLs(_ urls: [URL]) {
        let cleaned = urls.map { $0.standardizedFileURL }
        guard !cleaned.isEmpty else {
            statusText = "Drop a PDF, folder, or image files."
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            let prepared = try prepareDropInput(urls: cleaned)
            let newJobID = try bridge.createJob(inputDir: prepared.inputDir, outputDir: prepared.outputDir)
            self.jobID = newJobID
            loadJob()
            setPipelineStage(.ingest)
            statusText = "Created job \(newJobID) from dropped content. Start with Ingest."
        } catch {
            statusText = "Drop import failed: \(error.localizedDescription)"
        }
    }

    func setPipelineStage(_ stage: PipelineStage) {
        selectedPipelineStage = stage
        self.stage = stage.previewStage
    }

    func currentImagePath() -> String? {
        guard let page = selectedPage else { return nil }
        switch stage {
        case .normalized:
            return page.normalized_path
        case .inpaint:
            return page.output_paths["inpaint"] ?? page.normalized_path
        case .typeset:
            return page.output_paths["typeset"] ?? page.output_paths["inpaint"] ?? page.normalized_path
        case .exported:
            return page.output_paths["final_page"] ?? page.output_paths["compose"] ?? page.output_paths["typeset"] ?? page.normalized_path
        }
    }

    func loadSelectedPageState() {
        guard let page = selectedPage else {
            regions = []
            blocks = []
            return
        }

        regions = page.text_regions.map {
            EditableRegion(id: $0.id, text: $0.text, bbox: rectFromBBox($0.bbox), enabled: true)
        }
        blocks = page.typeset_blocks.map {
            EditableBlock(id: $0.region_id, text: $0.text, bbox: rectFromBBox($0.bbox), fontSize: $0.font_size, enabled: true)
        }

        selectedRegionID = regions.first?.id
        selectedBlockID = blocks.first?.id
    }

    func autoSelectStageForSelectedPage() {
        guard let page = selectedPage else { return }
        if let firstUnapproved = PipelineStage.allCases.first(where: { !isApproved($0, pageID: page.page_id) }) {
            setPipelineStage(firstUnapproved)
        } else {
            setPipelineStage(.export)
        }
    }

    func stageStatus(_ stage: PipelineStage) -> String {
        guard let page = selectedPage else { return "n/a" }
        return page.stages[stage.rawValue]?.status ?? "pending"
    }

    func isApproved(_ stage: PipelineStage, pageID: String) -> Bool {
        approvalMap[pageID]?[stage.rawValue] == true
    }

    func isCurrentStageApproved() -> Bool {
        guard let page = selectedPage else { return false }
        return isApproved(selectedPipelineStage, pageID: page.page_id)
    }

    func toggleApprovalForCurrentStage() {
        guard let page = selectedPage else { return }
        var pageMap = approvalMap[page.page_id] ?? [:]
        let key = selectedPipelineStage.rawValue
        let next = !(pageMap[key] == true)
        pageMap[key] = next
        approvalMap[page.page_id] = pageMap
        saveApprovals()
        statusText = next ? "Approved \(selectedPipelineStage.title) for page \(page.page_id)." : "Removed approval for \(selectedPipelineStage.title) on page \(page.page_id)."
    }

    func approveAndNext() {
        guard let page = selectedPage else { return }
        let approvedStage = selectedPipelineStage
        var pageMap = approvalMap[page.page_id] ?? [:]
        pageMap[approvedStage.rawValue] = true
        approvalMap[page.page_id] = pageMap
        saveApprovals()
        if let next = approvedStage.next {
            setPipelineStage(next)
        }
        statusText = "Approved \(approvedStage.title) for page \(page.page_id)."
    }

    private func canRunStage(stage: PipelineStage, pageIDs: [String]) -> Bool {
        guard let previous = stage.previous else {
            return true
        }
        let unapproved = pageIDs.filter { !isApproved(previous, pageID: $0) }
        if unapproved.isEmpty {
            return true
        }
        statusText = "Approve \(previous.title) before running \(stage.title). Missing: \(unapproved.joined(separator: ", "))."
        return false
    }

    func runCurrentStage() {
        guard !jobID.isEmpty else {
            statusText = "Enter a job id first."
            return
        }
        guard let page = selectedPage else {
            statusText = "Select a page first."
            return
        }

        let pageIDs = runStageForAllPages ? pages.map(\.page_id) : [page.page_id]
        let stageToRun = selectedPipelineStage
        if !canRunStage(stage: stageToRun, pageIDs: pageIDs) {
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            try bridge.runStage(
                jobID: jobID,
                pageID: runStageForAllPages ? nil : page.page_id,
                stage: stageToRun.rawValue,
                allPages: runStageForAllPages
            )
            loadJob()
            self.stage = stageToRun.previewStage
            statusText = runStageForAllPages
                ? "Ran stage \(stageToRun.title) for all pages."
                : "Ran stage \(stageToRun.title) for page \(page.page_id)."
        } catch {
            statusText = "Failed running stage: \(error.localizedDescription)"
        }
    }

    func toggleSelectedRegionEnabled() {
        guard let id = selectedRegionID, let idx = regions.firstIndex(where: { $0.id == id }) else { return }
        regions[idx].enabled.toggle()
    }

    func toggleSelectedBlockEnabled() {
        guard let id = selectedBlockID, let idx = blocks.firstIndex(where: { $0.id == id }) else { return }
        blocks[idx].enabled.toggle()
    }

    func applyEdits() {
        guard !jobID.isEmpty, let page = selectedPage else {
            statusText = "Select a page first."
            return
        }

        isBusy = true
        defer { isBusy = false }

        let regionPayload: [[String: Any]] = regions.map { region in
            [
                "region_id": region.id,
                "enabled": region.enabled,
                "bbox": bboxFromRect(region.bbox),
            ]
        }

        let blockPayload: [[String: Any]] = blocks.map { block in
            [
                "region_id": block.id,
                "enabled": block.enabled,
                "text": block.text,
                "bbox": bboxFromRect(block.bbox),
                "font_size": block.fontSize,
            ]
        }

        let payload: [String: Any] = [
            "pages": [
                page.page_id: [
                    "regions": regionPayload,
                    "blocks": blockPayload,
                ],
            ],
        ]

        do {
            try bridge.applyEdits(jobID: jobID, payload: payload)
            loadJob()
            statusText = "Applied edits and re-rendered page \(page.page_id)."
        } catch {
            statusText = "Failed applying edits: \(error.localizedDescription)"
        }
    }

    private static func findRepoRoot(start: URL) -> URL? {
        var current = start
        for _ in 0..<8 {
            let marker = current.appendingPathComponent("mtranslate/cli.py").path
            if FileManager.default.fileExists(atPath: marker) {
                return current
            }
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path {
                break
            }
            current = parent
        }
        return nil
    }

    private func prepareDropInput(urls: [URL]) throws -> (inputDir: URL, outputDir: URL) {
        let root = URL(fileURLWithPath: repoRoot)
        let token = Self.timestampToken()
        let inputDir = root
            .appendingPathComponent(".mtranslate_data")
            .appendingPathComponent("ui_inputs")
            .appendingPathComponent(token)
        let outputDir = root
            .appendingPathComponent("outputs")
            .appendingPathComponent("ui_jobs")
            .appendingPathComponent(token)
        try FileManager.default.createDirectory(at: inputDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        var imageFiles: [URL] = []
        var pdfFiles: [URL] = []
        for url in urls {
            collectInputs(at: url, images: &imageFiles, pdfs: &pdfFiles)
        }

        imageFiles = sortNaturally(imageFiles)
        pdfFiles = sortNaturally(pdfFiles)

        var pageIndex = 1
        for pdf in pdfFiles {
            pageIndex = try renderPDFPages(pdfURL: pdf, to: inputDir, startIndex: pageIndex)
        }
        for image in imageFiles {
            let ext = normalizedImageExtension(for: image)
            let dst = inputDir.appendingPathComponent(String(format: "%03d.%@", pageIndex, ext))
            if FileManager.default.fileExists(atPath: dst.path) {
                try FileManager.default.removeItem(at: dst)
            }
            try FileManager.default.copyItem(at: image, to: dst)
            pageIndex += 1
        }

        if pageIndex == 1 {
            throw NSError(domain: "MTranslateEditor", code: 11, userInfo: [
                NSLocalizedDescriptionKey: "No supported inputs found. Drop a PDF, folder, or image files.",
            ])
        }

        return (inputDir, outputDir)
    }

    private static func timestampToken() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return "\(fmt.string(from: Date()))_\(UUID().uuidString.prefix(6))"
    }

    private func collectInputs(at url: URL, images: inout [URL], pdfs: inout [URL]) {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return
        }

        if isDir.boolValue {
            guard let enumerator = FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else {
                return
            }
            for case let child as URL in enumerator {
                var childIsDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: child.path, isDirectory: &childIsDir), !childIsDir.boolValue else {
                    continue
                }
                if isPDFFile(child) {
                    pdfs.append(child)
                } else if isImageFile(child) {
                    images.append(child)
                }
            }
            return
        }

        if isPDFFile(url) {
            pdfs.append(url)
        } else if isImageFile(url) {
            images.append(url)
        }
    }

    private func sortNaturally(_ urls: [URL]) -> [URL] {
        urls.sorted { lhs, rhs in
            lhs.path.localizedStandardCompare(rhs.path) == .orderedAscending
        }
    }

    private func isPDFFile(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "pdf"
    }

    private func isImageFile(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return [
            "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "heic", "heif", "avif",
        ].contains(ext)
    }

    private func normalizedImageExtension(for url: URL) -> String {
        let ext = url.pathExtension.lowercased()
        if ext.isEmpty {
            return "png"
        }
        return ext
    }

    private func renderPDFPages(pdfURL: URL, to outputDir: URL, startIndex: Int) throws -> Int {
        guard let doc = CGPDFDocument(pdfURL as CFURL) else {
            throw NSError(domain: "MTranslateEditor", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "Failed to open PDF: \(pdfURL.lastPathComponent)",
            ])
        }
        var index = startIndex
        let pageCount = doc.numberOfPages
        let scale: CGFloat = 2.0

        for pageNum in 1...pageCount {
            guard let page = doc.page(at: pageNum) else { continue }
            let media = page.getBoxRect(.mediaBox)
            if media.width <= 0 || media.height <= 0 {
                continue
            }

            let width = max(1, Int((media.width * scale).rounded()))
            let height = max(1, Int((media.height * scale).rounded()))
            let cs = CGColorSpaceCreateDeviceRGB()
            guard let ctx = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                continue
            }

            ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))

            ctx.saveGState()
            ctx.translateBy(x: 0, y: CGFloat(height))
            ctx.scaleBy(x: scale, y: -scale)
            ctx.drawPDFPage(page)
            ctx.restoreGState()

            guard let cg = ctx.makeImage() else {
                continue
            }
            let dst = outputDir.appendingPathComponent(String(format: "%03d.png", index))
            guard let dest = CGImageDestinationCreateWithURL(dst as CFURL, "public.png" as CFString, 1, nil) else {
                continue
            }
            CGImageDestinationAddImage(dest, cg, nil)
            if !CGImageDestinationFinalize(dest) {
                throw NSError(domain: "MTranslateEditor", code: 13, userInfo: [
                    NSLocalizedDescriptionKey: "Failed writing rendered PDF page for \(pdfURL.lastPathComponent).",
                ])
            }
            index += 1
        }
        return index
    }
}
