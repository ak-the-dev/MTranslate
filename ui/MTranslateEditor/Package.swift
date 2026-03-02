// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MTranslateEditor",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "MTranslateEditor", targets: ["MTranslateEditor"]),
    ],
    targets: [
        .executableTarget(
            name: "MTranslateEditor",
            path: "Sources/MTranslateEditor"
        ),
    ]
)
