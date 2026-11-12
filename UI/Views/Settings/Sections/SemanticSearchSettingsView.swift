import SwiftUI
import Shared
import App

/// Settings card for semantic search (search by meaning).
///
/// The card holds no state of its own apart from its confirmation alert: every instance for a coordinator
/// (the Advanced tab, settings search) observes the same `SemanticSearchSettingsModel`, which goes on
/// downloading and turns the setting on when the card is no longer on screen.
struct SemanticSearchSettingsCard: View {
    @ObservedObject private var model: SemanticSearchSettingsModel
    @State private var isConfirmingIndexRemoval = false

    init(coordinator: AppCoordinator) {
        _model = ObservedObject(wrappedValue: SemanticSearchSettingsModel.shared(for: coordinator))
    }

    /// The card over a model made elsewhere (previews and tests).
    init(model: SemanticSearchSettingsModel) {
        _model = ObservedObject(wrappedValue: model)
    }

    var body: some View {
        ModernSettingsCard(title: "Semantic Search", icon: "sparkle.magnifyingglass") {
            VStack(alignment: .leading, spacing: 12) {
                ModernToggleRow(
                    title: "Search by meaning",
                    subtitle: "Find a moment by describing it, even when the exact words weren't on screen. "
                        + "Runs entirely on this Mac and makes Relevance the default search order. "
                        + "Downloads a \(model.textModelSizeMB) MB model once.",
                    isOn: Binding(get: { model.isEnabled }, set: { model.setEnabled($0) }),
                    disabled: !model.canChangeSwitches,
                    badge: "BETA"
                )

                ModernToggleRow(
                    title: "Also index what screens look like",
                    subtitle: "Finds a screen by what it showed (a chart, a photo, a video call) as well as by its "
                        + "text. Screens with no readable text at all aren't indexed yet. "
                        + "Downloads a \(model.imageModelSizeMB) MB image encoder once. Screenshots are indexed "
                        + "only while on power, about ten times more slowly than text.",
                    isOn: Binding(get: { model.isImagesEnabled }, set: { model.setImagesEnabled($0) }),
                    disabled: !model.canChangeSwitches || !model.isEnabled
                )

                if model.downloading != nil {
                    ProgressView(value: model.downloadProgress)
                        .tint(.retraceAccent)
                }

                if let statusText = model.statusText {
                    Text(statusText)
                        .font(.retraceCaption2)
                        .foregroundColor(.retraceSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if let errorMessage = model.errorMessage {
                    Text(errorMessage)
                        .font(.retraceCaption2)
                        .foregroundColor(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if model.hasIndex && model.downloading == nil {
                    HStack {
                        Spacer()
                        ModernButton(title: "Remove Index", icon: "trash", style: .secondary) {
                            isConfirmingIndexRemoval = true
                        }
                        .disabled(model.isRemovingIndex)
                    }
                }
            }
        }
        .alert("Remove Semantic Search Index?", isPresented: $isConfirmingIndexRemoval) {
            Button("Cancel", role: .cancel) {}
            Button("Remove Index", role: .destructive) {
                model.removeIndex()
            }
        } message: {
            Text(model.removeIndexConfirmation)
        }
        .onAppear { model.cardDidAppear() }
        .onDisappear { model.cardDidDisappear() }
    }
}
