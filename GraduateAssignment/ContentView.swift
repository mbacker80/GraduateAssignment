import SwiftUI
import CoreML
import Vision
import UIKit

class ImageClassifier: ObservableObject {
    @Published var fastViTResult: String = ""
    @Published var resnet50Result: String = ""
    @Published var mobileNetV2Result: String = ""
    
    private var fastViTModel: VNCoreMLModel?
    private var resnet50Model: VNCoreMLModel?
    private var mobileNetV2Model: VNCoreMLModel?

    init() {
        loadModels()
    }
    
    func loadModels() {
        do {
            fastViTModel = try VNCoreMLModel(for: FastViTMA36F16().model)
            resnet50Model = try VNCoreMLModel(for: Resnet50().model)
            mobileNetV2Model = try VNCoreMLModel(for: MobileNetV2().model)
        } catch {
            print("Error loading models: \(error)")
        }
    }
    
    func classify(image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        classifyWith(model: fastViTModel, handler: handler) { result in
            DispatchQueue.main.async { self.fastViTResult = "FastViT: \(result)" }
        }
        classifyWith(model: resnet50Model, handler: handler) { result in
            DispatchQueue.main.async { self.resnet50Result = "ResNet50: \(result)" }
        }
        classifyWith(model: mobileNetV2Model, handler: handler) { result in
            DispatchQueue.main.async { self.mobileNetV2Result = "MobileNetV2: \(result)" }
        }
    }
    
    private func classifyWith(model: VNCoreMLModel?, handler: VNImageRequestHandler, completion: @escaping (String) -> Void) {
        guard let model = model else {
            completion("Model unavailable")
            return
        }
        
        let request = VNCoreMLRequest(model: model) { req, err in
            if let results = req.results as? [VNClassificationObservation],
               let top = results.first {
                completion("\(top.identifier) (\(String(format: "%.2f", top.confidence * 100))%)")
            } else {
                completion("No result")
            }
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                completion("Error: \(error.localizedDescription)")
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var classifier = ImageClassifier()
    @State private var selectedImage: UIImage? = nil
    @State private var isPickerPresented = false

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Image Classifier")
                    .font(.largeTitle)
                    .padding(.top)

                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 300, maxHeight: 300)
                        .cornerRadius(10)
                } else {
                    Text("No image selected")
                        .foregroundColor(.gray)
                }

                Button("Select Image") {
                    isPickerPresented = true
                }
                .buttonStyle(.borderedProminent)

                if selectedImage != nil {
                    Button("Classify Image") {
                        if let image = selectedImage {
                            classifier.classify(image: image)
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(.green)
                }

                VStack(alignment: .leading, spacing: 10) {
                    if !classifier.fastViTResult.isEmpty {
                        Text(classifier.fastViTResult)
                    }
                    if !classifier.resnet50Result.isEmpty {
                        Text(classifier.resnet50Result)
                    }
                    if !classifier.mobileNetV2Result.isEmpty {
                        Text(classifier.mobileNetV2Result)
                    }
                }
                .padding()

                Spacer()
            }
            .sheet(isPresented: $isPickerPresented) {
                ImagePicker(image: $selectedImage)
            }
            .padding()
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            picker.dismiss(animated: true)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}

#Preview {
    ContentView()
}
