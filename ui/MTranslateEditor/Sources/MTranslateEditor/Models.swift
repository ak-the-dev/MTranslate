import Foundation
import CoreGraphics

struct JobManifest: Decodable {
    let job_id: String
    let pages: [String: PageManifest]
}

struct StageState: Decodable {
    let name: String
    let status: String
}

struct PageManifest: Decodable {
    let page_id: String
    let index: Int
    let normalized_path: String?
    let width: Int?
    let height: Int?
    let stages: [String: StageState]
    let output_paths: [String: String]
    let text_regions: [TextRegion]
    let typeset_blocks: [TypesetBlock]
}

struct TextRegion: Decodable {
    let id: String
    let text: String
    let bbox: [Int]
}

struct TypesetBlock: Decodable {
    let region_id: String
    let text: String
    let bbox: [Int]
    let font_size: Int
}

struct EditableRegion: Identifiable {
    let id: String
    var text: String
    var bbox: CGRect
    var enabled: Bool
}

struct EditableBlock: Identifiable {
    let id: String
    var text: String
    var bbox: CGRect
    var fontSize: Int
    var enabled: Bool
}

enum PreviewStage: String, CaseIterable, Identifiable {
    case normalized
    case inpaint
    case typeset
    case exported

    var id: String { rawValue }

    var title: String {
        switch self {
        case .normalized:
            return "Normalized"
        case .inpaint:
            return "Inpaint"
        case .typeset:
            return "Typeset"
        case .exported:
            return "Exported"
        }
    }
}

enum OverlayMode: String, CaseIterable, Identifiable {
    case regions
    case blocks

    var id: String { rawValue }

    var title: String {
        switch self {
        case .regions:
            return "Infill Regions"
        case .blocks:
            return "Text Layout"
        }
    }
}

enum PipelineStage: String, CaseIterable, Identifiable {
    case ingest
    case ocr
    case vlm_refine
    case semantic_group
    case translate
    case mask
    case inpaint
    case typeset
    case compose
    case export

    var id: String { rawValue }

    var title: String {
        switch self {
        case .ingest:
            return "Ingest"
        case .ocr:
            return "OCR"
        case .vlm_refine:
            return "VLM"
        case .semantic_group:
            return "Semantic"
        case .translate:
            return "Translate"
        case .mask:
            return "Mask"
        case .inpaint:
            return "Inpaint"
        case .typeset:
            return "Typeset"
        case .compose:
            return "Compose"
        case .export:
            return "Export"
        }
    }

    var previewStage: PreviewStage {
        switch self {
        case .ingest, .ocr, .vlm_refine, .semantic_group, .translate, .mask:
            return .normalized
        case .inpaint:
            return .inpaint
        case .typeset, .compose:
            return .typeset
        case .export:
            return .exported
        }
    }

    var previous: PipelineStage? {
        guard let idx = PipelineStage.allCases.firstIndex(of: self), idx > 0 else {
            return nil
        }
        return PipelineStage.allCases[idx - 1]
    }

    var next: PipelineStage? {
        guard let idx = PipelineStage.allCases.firstIndex(of: self), idx < (PipelineStage.allCases.count - 1) else {
            return nil
        }
        return PipelineStage.allCases[idx + 1]
    }
}

func rectFromBBox(_ bbox: [Int]) -> CGRect {
    guard bbox.count == 4 else { return .zero }
    return CGRect(x: bbox[0], y: bbox[1], width: max(1, bbox[2]), height: max(1, bbox[3]))
}

func bboxFromRect(_ rect: CGRect) -> [Int] {
    [
        max(0, Int(rect.origin.x.rounded())),
        max(0, Int(rect.origin.y.rounded())),
        max(1, Int(rect.size.width.rounded())),
        max(1, Int(rect.size.height.rounded())),
    ]
}
