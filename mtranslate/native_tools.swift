import Foundation
import AppKit
import Vision
import CoreGraphics
import CoreText

enum NativeError: Error {
    case message(String)
}

func fail(_ message: String) -> Never {
    fputs("error: \(message)\n", stderr)
    exit(1)
}

func loadJSON(path: String) -> [String: Any] {
    guard let data = FileManager.default.contents(atPath: path) else {
        fail("cannot read request json: \(path)")
    }
    do {
        let any = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = any as? [String: Any] else {
            fail("request json must be an object")
        }
        return dict
    } catch {
        fail("invalid request json: \(error)")
    }
}

func writeJSON(path: String, payload: [String: Any]) {
    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    } catch {
        fail("failed writing response json: \(error)")
    }
}

func cgImage(from path: String) -> CGImage? {
    guard let image = NSImage(contentsOfFile: path) else { return nil }
    var rect = CGRect(origin: .zero, size: image.size)
    return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
}

func recognizedTextRegions(cg: CGImage, width: Int, height: Int) -> [[String: Any]] {
    let req = VNRecognizeTextRequest()
    req.usesLanguageCorrection = false
    req.recognitionLevel = .accurate
    req.minimumTextHeight = 0.01
    if #available(macOS 13.0, *) {
        req.automaticallyDetectsLanguage = true
    } else {
        req.recognitionLanguages = ["ja-JP", "en-US"]
    }

    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])

    var regions: [[String: Any]] = []
    if let observations = req.results as? [VNRecognizedTextObservation] {
        for (index, obs) in observations.enumerated() {
            guard let top = obs.topCandidates(1).first else { continue }
            let text = top.string.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }

            let bb = obs.boundingBox
            let x = Int(bb.origin.x * CGFloat(width))
            let y = Int((1.0 - bb.origin.y - bb.height) * CGFloat(height))
            let w = Int(bb.width * CGFloat(width))
            let h = Int(bb.height * CGFloat(height))
            if w <= 2 || h <= 2 { continue }

            let orientation = h > w ? "vertical" : "horizontal"
            let poly = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
            regions.append([
                "id": "ocr_\(index)",
                "text": text,
                "bbox": [x, y, w, h],
                "polygon": poly,
                "orientation": orientation,
                "confidence": Double(top.confidence),
                "source": "vision_recognize",
            ])
        }
    }
    return regions
}

func rectangleTextRegions(cg: CGImage, width: Int, height: Int) -> [[String: Any]] {
    let req = VNDetectTextRectanglesRequest()
    req.reportCharacterBoxes = false
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([req])

    var regions: [[String: Any]] = []
    if let boxes = req.results as? [VNTextObservation] {
        for (index, obs) in boxes.enumerated() {
            let bb = obs.boundingBox
            let x = Int(bb.origin.x * CGFloat(width))
            let y = Int((1.0 - bb.origin.y - bb.height) * CGFloat(height))
            let w = Int(bb.width * CGFloat(width))
            let h = Int(bb.height * CGFloat(height))
            if w <= 2 || h <= 2 { continue }

            let orientation = h > w ? "vertical" : "horizontal"
            let poly = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
            regions.append([
                "id": "det_\(index)",
                "text": "__UNRESOLVED_REGION_\(index + 1)__",
                "bbox": [x, y, w, h],
                "polygon": poly,
                "orientation": orientation,
                "confidence": 0.15,
                "source": "vision_detect",
            ])
        }
    }

    return regions
}

func ocr(images: [String]) -> [String: Any] {
    var result: [String: Any] = [:]
    for imagePath in images {
        guard let cg = cgImage(from: imagePath) else {
            result[imagePath] = []
            continue
        }

        let width = cg.width
        let height = cg.height

        var regions = recognizedTextRegions(cg: cg, width: width, height: height)
        if regions.isEmpty {
            regions = rectangleTextRegions(cg: cg, width: width, height: height)
        }
        result[imagePath] = regions
    }

    return ["regions": result]
}

func ensureFontRegistered(path: String) {
    let url = URL(fileURLWithPath: path)
    var err: Unmanaged<CFError>?
    _ = CTFontManagerRegisterFontsForURL(url as CFURL, .process, &err)
}

func bestFont(size: CGFloat, preferredFamily: String?) -> NSFont {
    if let family = preferredFamily, let f = NSFont(name: family, size: size) {
        return f
    }
    if let f = NSFont(name: "Comic Neue", size: size) {
        return f
    }
    if let f = NSFont(name: "Helvetica Neue", size: size) {
        return f
    }
    return NSFont.systemFont(ofSize: size)
}

func fitText(
    _ text: String,
    in rect: CGRect,
    preferredFamily: String?,
    vertical: Bool,
    preferredSize: CGFloat?
) -> (NSAttributedString, CGFloat) {
    let hasLatin = text.range(of: "[A-Za-z]", options: .regularExpression) != nil
    let content: String
    if vertical && !hasLatin {
        content = text.map { String($0) }.joined(separator: "\n")
    } else {
        content = text
    }

    let style = NSMutableParagraphStyle()
    style.alignment = .center
    style.lineBreakMode = .byWordWrapping

    let maxSize = max(8.0, preferredSize ?? (min(rect.width, rect.height) * 0.45))
    var size = maxSize

    while size >= 8 {
        let font = bestFont(size: size, preferredFamily: preferredFamily)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.black,
            .paragraphStyle: style,
        ]
        let attributed = NSAttributedString(string: content, attributes: attrs)
        if hasLatin && !content.contains(" ") {
            let oneLine = attributed.size()
            if oneLine.width <= rect.width * 0.96 && oneLine.height <= rect.height {
                return (attributed, size)
            }
            size -= 1
            continue
        }
        let bounds = attributed.boundingRect(
            with: NSSize(width: rect.width, height: CGFloat.greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading]
        )
        if bounds.height <= rect.height && bounds.width <= rect.width {
            return (attributed, size)
        }
        size -= 1
    }

    let fallback = NSAttributedString(
        string: content,
        attributes: [
            .font: bestFont(size: max(8, min(10, maxSize)), preferredFamily: preferredFamily),
            .foregroundColor: NSColor.black,
            .paragraphStyle: style,
        ]
    )
    return (fallback, max(8, min(10, maxSize)))
}

func typeset(tasks: [[String: Any]], fontPath: String?) -> [String: Any] {
    if let fp = fontPath {
        ensureFontRegistered(path: fp)
    }

    var outputs: [[String: Any]] = []

    for task in tasks {
        guard
            let input = task["input"] as? String,
            let output = task["output"] as? String
        else {
            continue
        }

        guard let source = NSImage(contentsOfFile: input) else {
            outputs.append(["input": input, "output": output, "status": "failed", "error": "image_open_failed"])
            continue
        }

        guard let cg = cgImage(from: input) else {
            outputs.append(["input": input, "output": output, "status": "failed", "error": "cgimage_failed"])
            continue
        }

        let width = cg.width
        let height = cg.height

        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else {
            outputs.append(["input": input, "output": output, "status": "failed", "error": "bitmap_alloc_failed"])
            continue
        }

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

        let canvas = CGRect(x: 0, y: 0, width: width, height: height)
        source.draw(in: canvas)

        let blocks = (task["blocks"] as? [[String: Any]]) ?? []
        for block in blocks {
            guard
                let bbox = block["bbox"] as? [Int],
                bbox.count == 4,
                let text = block["text"] as? String
            else {
                continue
            }
            let x = bbox[0]
            let y = bbox[1]
            let w = max(1, bbox[2])
            let h = max(1, bbox[3])

            let drawY = height - y - h
            let rect = CGRect(x: x, y: drawY, width: w, height: h)
            let inset = max(2.0, min(rect.width, rect.height) * 0.08)
            let textRectBase = rect.insetBy(dx: inset, dy: inset)
            let textRect = textRectBase.width > 1 && textRectBase.height > 1 ? textRectBase : rect

            let orientation = (block["orientation"] as? String) ?? "horizontal"
            let vertical = orientation == "vertical"
            let family = block["font_family"] as? String
            let fontSize = (block["font_size"] as? Int).map { CGFloat($0) }
            let (attributed, _) = fitText(
                text,
                in: textRect,
                preferredFamily: family,
                vertical: vertical,
                preferredSize: fontSize
            )
            let singleTokenLatin = text.range(of: "^[A-Za-z]+[\\.!?]?$", options: .regularExpression) != nil
            if singleTokenLatin {
                let one = attributed.size()
                let drawPoint = NSPoint(
                    x: textRect.origin.x + max(0, (textRect.width - one.width) / 2.0),
                    y: textRect.origin.y + max(0, (textRect.height - one.height) / 2.0)
                )
                attributed.draw(at: drawPoint)
            } else {
                let bounds = attributed.boundingRect(
                    with: NSSize(width: textRect.width, height: CGFloat.greatestFiniteMagnitude),
                    options: [.usesLineFragmentOrigin, .usesFontLeading]
                )
                let drawRect = CGRect(
                    x: textRect.origin.x,
                    y: textRect.origin.y + max(0, (textRect.height - bounds.height) / 2.0),
                    width: textRect.width,
                    height: textRect.height
                )
                attributed.draw(with: drawRect, options: [.usesLineFragmentOrigin, .usesFontLeading])
            }
        }

        NSGraphicsContext.restoreGraphicsState()

        let outURL = URL(fileURLWithPath: output)
        do {
            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard let pngData = rep.representation(using: .png, properties: [:]) else {
                outputs.append(["input": input, "output": output, "status": "failed", "error": "png_encode_failed"])
                continue
            }
            try pngData.write(to: outURL)
            outputs.append(["input": input, "output": output, "status": "done"])
        } catch {
            outputs.append(["input": input, "output": output, "status": "failed", "error": "\(error)"])
        }
    }

    return ["tasks": outputs]
}

func imagesToPDF(images: [String], output: String) -> [String: Any] {
    let url = URL(fileURLWithPath: output)
    do {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    } catch {
        return ["status": "failed", "error": "mkdir_failed"]
    }

    guard let consumer = CGDataConsumer(url: url as CFURL) else {
        return ["status": "failed", "error": "consumer_failed"]
    }

    var mediaBox = CGRect(x: 0, y: 0, width: 100, height: 100)
    guard let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
        return ["status": "failed", "error": "context_failed"]
    }

    var count = 0
    for path in images {
        guard let cg = cgImage(from: path) else { continue }
        var box = CGRect(x: 0, y: 0, width: cg.width, height: cg.height)
        let pageInfo = [kCGPDFContextMediaBox as String: box] as CFDictionary
        ctx.beginPDFPage(pageInfo)
        ctx.draw(cg, in: box)
        ctx.endPDFPage()
        count += 1
    }

    ctx.closePDF()
    return ["status": "done", "pages": count, "output": output]
}

if CommandLine.arguments.count < 3 {
    fail("usage: swift native_tools.swift <request.json> <response.json>")
}

let requestPath = CommandLine.arguments[1]
let responsePath = CommandLine.arguments[2]
let request = loadJSON(path: requestPath)
let command = request["command"] as? String ?? ""

let response: [String: Any]
switch command {
case "ocr":
    let images = request["images"] as? [String] ?? []
    response = ocr(images: images)
case "typeset":
    let tasks = request["tasks"] as? [[String: Any]] ?? []
    let fontPath = request["font_path"] as? String
    response = typeset(tasks: tasks, fontPath: fontPath)
case "images_to_pdf":
    let images = request["images"] as? [String] ?? []
    let output = request["output"] as? String ?? ""
    response = imagesToPDF(images: images, output: output)
default:
    response = ["status": "failed", "error": "unknown_command"]
}

writeJSON(path: responsePath, payload: response)
