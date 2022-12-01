//
//  PoseEstimator.swift
//  PoseEstimationDemo
//
//  Created by Kevin Beaulieu on 2022-12-01.
//

import AVFoundation
import CoreML
import UIKit
import VideoToolbox

class PoseEstimator: NSObject {
    let captureSession = AVCaptureSession()
    private let processingQueue = DispatchQueue(
        label: "com.kevinmbeaulieu.PoseEstimatorDemo.PoseEstimator",
        qos: .userInteractive
    )
    private let model: PoseNetMobileNet075S8FP16 = {
        try! PoseNetMobileNet075S8FP16(configuration: MLModelConfiguration())
    }()

    override init() {
        super.init()

        setUpCaptureSession()
    }

    func start() {
        Task {
            guard !captureSession.isRunning else { return }
            captureSession.startRunning()
        }
    }

    private func setUpCaptureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        guard let camera = AVCaptureDevice.default(for: .video),
              let cameraInput = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(cameraInput)
        else {
            print("Failed to add camera input")
            return
        }
        captureSession.addInput(cameraInput)

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        guard captureSession.canAddOutput(videoOutput) else {
            print("Failed to add video output")
            return
        }
        captureSession.addOutput(videoOutput)

        captureSession.commitConfiguration()
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        assert(processingQueue.label == String(cString: __dispatch_queue_get_label(nil), encoding: .utf8))

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly).assertReturnSuccess()
        let pixelBufferSize = (
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer)
        )
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly).assertReturnSuccess()
        let modelInputSize = (width: 513, height: 513)
        let cropRect = CGRect(
            x: (pixelBufferSize.width - modelInputSize.width) / 2,
            y: (pixelBufferSize.height - modelInputSize.height) / 2,
            width: modelInputSize.width,
            height: modelInputSize.height
        )
        guard let croppedPixelBuffer = pixelBuffer.crop(to: cropRect) else {
            print("Failed to make CGImage")
            return
        }

        // TODO: Map croppedPixelBuffer from kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange to kCVPixelFormatType_32BGRA ?

        let input = PoseNetMobileNet075S8FP16Input(image: croppedPixelBuffer)

        guard let output = try? model.prediction(input: input) else {
            print("Failed to predict pose in frame")
            return
        }

        print(output)
    }
}

extension PoseEstimator: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = sampleBuffer.imageBuffer else { return }

        processFrame(pixelBuffer)
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        print("***Dropped frame***")
    }
}
