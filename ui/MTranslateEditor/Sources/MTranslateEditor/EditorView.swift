import SwiftUI
import AppKit
import UniformTypeIdentifiers

private struct CanvasTransform {
    let imageSize: CGSize
    let viewSize: CGSize

    var scale: CGFloat {
        max(0.0001, min(viewSize.width / imageSize.width, viewSize.height / imageSize.height))
    }

    var drawSize: CGSize {
        CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }

    var origin: CGPoint {
        CGPoint(x: (viewSize.width - drawSize.width) * 0.5, y: (viewSize.height - drawSize.height) * 0.5)
    }

    func toViewRect(_ rect: CGRect) -> CGRect {
        CGRect(
            x: origin.x + rect.origin.x * scale,
            y: origin.y + rect.origin.y * scale,
            width: rect.width * scale,
            height: rect.height * scale
        )
    }

    func deltaToImage(_ delta: CGSize) -> CGSize {
        CGSize(width: delta.width / scale, height: delta.height / scale)
    }

    func clamp(rect: CGRect) -> CGRect {
        var r = rect
        r.size.width = max(8, min(imageSize.width, r.size.width))
        r.size.height = max(8, min(imageSize.height, r.size.height))
        r.origin.x = max(0, min(imageSize.width - r.size.width, r.origin.x))
        r.origin.y = max(0, min(imageSize.height - r.size.height, r.origin.y))
        return r
    }
}

private struct BlockOverlayView: View {
    @Binding var block: EditableBlock
    let transform: CanvasTransform
    let isSelected: Bool
    let onSelect: () -> Void

    @State private var dragStart: CGRect?
    @State private var resizeStart: CGRect?

    var body: some View {
        let viewRect = transform.toViewRect(block.bbox)

        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 4)
                .stroke(isSelected ? Color.blue : Color.white, lineWidth: isSelected ? 2 : 1)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill((isSelected ? Color.blue : Color.gray).opacity(0.15))
                )

            Text(block.text)
                .font(.system(size: max(9, CGFloat(block.fontSize) * 0.4)))
                .foregroundColor(.white)
                .padding(4)
                .lineLimit(2)
        }
        .frame(width: viewRect.width, height: viewRect.height)
        .position(x: viewRect.midX, y: viewRect.midY)
        .contentShape(Rectangle())
        .onTapGesture {
            onSelect()
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    onSelect()
                    if dragStart == nil {
                        dragStart = block.bbox
                    }
                    guard let start = dragStart else { return }
                    let delta = transform.deltaToImage(value.translation)
                    var next = start
                    next.origin.x += delta.width
                    next.origin.y += delta.height
                    block.bbox = transform.clamp(rect: next)
                }
                .onEnded { _ in
                    dragStart = nil
                }
        )
        .overlay(alignment: .bottomTrailing) {
            if isSelected {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 10, height: 10)
                    .padding(2)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                if resizeStart == nil {
                                    resizeStart = block.bbox
                                }
                                guard let start = resizeStart else { return }
                                let delta = transform.deltaToImage(value.translation)
                                var next = start
                                next.size.width = max(8, start.width + delta.width)
                                next.size.height = max(8, start.height + delta.height)
                                block.bbox = transform.clamp(rect: next)
                            }
                            .onEnded { _ in
                                resizeStart = nil
                            }
                    )
            }
        }
    }
}

private struct RegionOverlayView: View {
    let region: EditableRegion
    let transform: CanvasTransform
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        let viewRect = transform.toViewRect(region.bbox)
        RoundedRectangle(cornerRadius: 4)
            .stroke(region.enabled ? (isSelected ? Color.green : Color.yellow) : Color.red, style: StrokeStyle(lineWidth: isSelected ? 2 : 1, dash: region.enabled ? [] : [6, 4]))
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill((region.enabled ? Color.green : Color.red).opacity(0.15))
            )
            .frame(width: viewRect.width, height: viewRect.height)
            .position(x: viewRect.midX, y: viewRect.midY)
            .contentShape(Rectangle())
            .onTapGesture {
                onSelect()
            }
    }
}

private struct PageCanvasView: View {
    @Binding var regions: [EditableRegion]
    @Binding var blocks: [EditableBlock]
    @Binding var selectedRegionID: String?
    @Binding var selectedBlockID: String?
    let mode: OverlayMode
    let imagePath: String?
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let transform = CanvasTransform(imageSize: imageSize, viewSize: geo.size)
            ZStack(alignment: .topLeading) {
                Color.black.opacity(0.12)

                if let imagePath, let image = NSImage(contentsOfFile: imagePath) {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: transform.drawSize.width, height: transform.drawSize.height)
                        .position(x: transform.origin.x + transform.drawSize.width * 0.5, y: transform.origin.y + transform.drawSize.height * 0.5)
                }

                if mode == .regions {
                    ForEach(regions) { region in
                        RegionOverlayView(
                            region: region,
                            transform: transform,
                            isSelected: selectedRegionID == region.id,
                            onSelect: {
                                selectedRegionID = region.id
                            }
                        )
                    }
                } else {
                    ForEach(Array(blocks.enumerated()), id: \.element.id) { idx, _ in
                        BlockOverlayView(
                            block: $blocks[idx],
                            transform: transform,
                            isSelected: selectedBlockID == blocks[idx].id,
                            onSelect: {
                                selectedBlockID = blocks[idx].id
                            }
                        )
                    }
                }
            }
        }
    }
}

struct EditorView: View {
    @StateObject private var vm = EditorViewModel()

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                Text("Repo")
                TextField("/path/to/MTranslate", text: $vm.repoRoot)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 320)
                Text("Job")
                TextField("20260225_...", text: $vm.jobID)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 220)
                Button("Load") {
                    vm.loadJob()
                }
                .disabled(vm.isBusy)

                Button("Reload") {
                    vm.reloadPage()
                }
                .disabled(vm.isBusy || vm.jobID.isEmpty)

                Spacer()

                Picker("Preview", selection: $vm.stage) {
                    ForEach(PreviewStage.allCases) { stage in
                        Text(stage.title).tag(stage)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 320)
            }

            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(vm.isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.5), style: StrokeStyle(lineWidth: vm.isDropTargeted ? 2 : 1, dash: [8, 6]))
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(vm.isDropTargeted ? Color.accentColor.opacity(0.08) : Color.clear)
                    )
                Text("Drop PDFs, folders, or images here to create a new job")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .frame(height: 64)
            .onDrop(of: [UTType.fileURL.identifier], isTargeted: $vm.isDropTargeted) { providers in
                handleDrop(providers: providers)
            }

            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading) {
                    Text("Pages")
                        .font(.headline)
                    List {
                        ForEach(vm.pages, id: \.page_id) { page in
                            Button {
                                vm.selectedPageID = page.page_id
                                vm.loadSelectedPageState()
                                vm.autoSelectStageForSelectedPage()
                            } label: {
                                Text(page.page_id)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.vertical, 2)
                            }
                            .buttonStyle(.plain)
                            .listRowBackground(vm.selectedPageID == page.page_id ? Color.blue.opacity(0.2) : Color.clear)
                        }
                    }
                    .frame(minWidth: 90, maxWidth: 110)
                }

                PageCanvasView(
                    regions: $vm.regions,
                    blocks: $vm.blocks,
                    selectedRegionID: $vm.selectedRegionID,
                    selectedBlockID: $vm.selectedBlockID,
                    mode: vm.overlayMode,
                    imagePath: vm.currentImagePath(),
                    imageSize: vm.imageSize
                )
                .frame(minWidth: 760, minHeight: 920)

                VStack(alignment: .leading, spacing: 10) {
                    Text("Stage Approval")
                        .font(.headline)

                    Toggle("Run Stage on All Pages", isOn: $vm.runStageForAllPages)
                        .toggleStyle(.checkbox)

                    List {
                        ForEach(PipelineStage.allCases) { pipelineStage in
                            Button {
                                vm.setPipelineStage(pipelineStage)
                            } label: {
                                HStack {
                                    Text(pipelineStage.title)
                                    Spacer()
                                    Text(vm.stageStatus(pipelineStage))
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                    if !vm.selectedPageID.isEmpty {
                                        Image(systemName: vm.isApproved(pipelineStage, pageID: vm.selectedPageID) ? "checkmark.circle.fill" : "circle")
                                            .foregroundColor(vm.isApproved(pipelineStage, pageID: vm.selectedPageID) ? .green : .secondary)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .buttonStyle(.plain)
                            .listRowBackground(vm.selectedPipelineStage == pipelineStage ? Color.orange.opacity(0.2) : Color.clear)
                        }
                    }
                    .frame(width: 260, height: 220)

                    HStack {
                        Button("Run Stage") {
                            vm.runCurrentStage()
                        }
                        .disabled(vm.isBusy || vm.selectedPageID.isEmpty)

                        Button(vm.isCurrentStageApproved() ? "Unapprove" : "Approve") {
                            vm.toggleApprovalForCurrentStage()
                        }
                        .disabled(vm.selectedPageID.isEmpty)
                    }

                    Button("Approve + Next") {
                        vm.approveAndNext()
                    }
                    .disabled(vm.selectedPageID.isEmpty)

                    Picker("Mode", selection: $vm.overlayMode) {
                        ForEach(OverlayMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    if vm.overlayMode == .regions {
                        Text("Regions")
                            .font(.headline)
                        List {
                            ForEach(vm.regions) { region in
                                Button {
                                    vm.selectedRegionID = region.id
                                } label: {
                                    HStack {
                                        Image(systemName: region.enabled ? "checkmark.square.fill" : "square")
                                            .foregroundColor(region.enabled ? .green : .red)
                                        Text(region.id)
                                            .font(.caption)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .buttonStyle(.plain)
                                .listRowBackground(vm.selectedRegionID == region.id ? Color.green.opacity(0.2) : Color.clear)
                            }
                        }
                        .frame(width: 260, height: 360)

                        Button("Toggle Region Enabled") {
                            vm.toggleSelectedRegionEnabled()
                        }
                    } else {
                        Text("Text Blocks")
                            .font(.headline)
                        List {
                            ForEach(vm.blocks) { block in
                                Button {
                                    vm.selectedBlockID = block.id
                                } label: {
                                    HStack {
                                        Image(systemName: block.enabled ? "checkmark.square.fill" : "square")
                                            .foregroundColor(block.enabled ? .green : .red)
                                        Text(block.id)
                                            .font(.caption)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .buttonStyle(.plain)
                                .listRowBackground(vm.selectedBlockID == block.id ? Color.blue.opacity(0.2) : Color.clear)
                            }
                        }
                        .frame(width: 260, height: 260)

                        Button("Toggle Block Enabled") {
                            vm.toggleSelectedBlockEnabled()
                        }

                        if let selected = vm.selectedBlockID,
                           let idx = vm.blocks.firstIndex(where: { $0.id == selected }) {
                            Text("Text")
                                .font(.headline)
                            TextEditor(text: $vm.blocks[idx].text)
                                .font(.system(size: 13))
                                .frame(width: 260, height: 120)
                                .border(Color.gray.opacity(0.4))

                            Stepper(value: $vm.blocks[idx].fontSize, in: 6 ... 120) {
                                Text("Font Size: \(vm.blocks[idx].fontSize)")
                            }
                        }
                    }

                    Button("Apply Edits + Re-render Page") {
                        vm.applyEdits()
                    }
                    .disabled(vm.isBusy || vm.selectedPageID.isEmpty)

                    Text(vm.statusText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 260, alignment: .leading)

                    Spacer()
                }
                .frame(width: 280)
            }
        }
        .padding(10)
        .task {
            vm.initializeRuntimeOnLaunch()
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        let fileType = UTType.fileURL.identifier
        let candidates = providers.filter { $0.hasItemConformingToTypeIdentifier(fileType) }
        if candidates.isEmpty {
            return false
        }

        let group = DispatchGroup()
        let lock = NSLock()
        var urls: [URL] = []

        for provider in candidates {
            group.enter()
            provider.loadItem(forTypeIdentifier: fileType, options: nil) { item, _ in
                defer { group.leave() }

                var resolved: URL?
                if let data = item as? Data {
                    resolved = NSURL(absoluteURLWithDataRepresentation: data, relativeTo: nil) as URL?
                } else if let nsurl = item as? NSURL {
                    resolved = nsurl as URL
                } else if let text = item as? String {
                    resolved = URL(string: text)
                }

                if let resolved {
                    lock.lock()
                    urls.append(resolved)
                    lock.unlock()
                }
            }
        }

        group.notify(queue: .main) {
            vm.importDroppedURLs(urls)
        }
        return true
    }
}
