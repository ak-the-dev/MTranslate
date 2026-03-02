import SwiftUI

@main
struct MTranslateEditorApp: App {
    var body: some Scene {
        WindowGroup {
            EditorView()
                .frame(minWidth: 1300, minHeight: 980)
        }
    }
}
