//
//  CameraPreviewView.swift
//  PoseEstimationDemo
//
//  Created by Kevin Beaulieu on 2022-12-02.
//

import AVFoundation
import SwiftUI
import UIKit

final class CameraPreviewUIView: UIView {
    private let previewLayer: AVCaptureVideoPreviewLayer = {
        let previewLayer = AVCaptureVideoPreviewLayer()
        previewLayer.videoGravity = .resizeAspectFill
        return previewLayer
    }()

    init() {
        super.init(frame: .zero)

        layer.addSublayer(previewLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        previewLayer.frame = layer.bounds
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func setCaptureSession(_ session: AVCaptureSession?) {
        previewLayer.session = session
    }
}

struct CameraPreviewView: View, UIViewRepresentable {
    @Binding var session: AVCaptureSession?

    func makeUIView(context: Context) -> CameraPreviewUIView {
        CameraPreviewUIView()
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.setCaptureSession(session)
    }
}
