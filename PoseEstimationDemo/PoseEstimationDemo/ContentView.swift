//
//  ContentView.swift
//  PoseEstimationDemo
//
//  Created by Kevin Beaulieu on 2022-12-01.
//

import AVFoundation
import SwiftUI

struct ContentView: View {
    private let poseEstimator = PoseEstimator()

    @State private var captureSession: AVCaptureSession?

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            Text("Hello, world!")
            CameraPreviewView(session: $captureSession)
                .frame(width: 300, height: 300)
                .background(Color.blue)
        }
        .padding()
        .onAppear {
            captureSession = poseEstimator.captureSession
            poseEstimator.start()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
