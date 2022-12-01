//
//  CVPixelBuffer+Extensions.swift
//  PoseEstimationDemo
//
//  Created by Kevin Beaulieu on 2022-12-02.
//

import Accelerate
import CoreGraphics
import CoreVideo
import VideoToolbox

extension CVPixelBuffer {
    func makeCGImage() -> CGImage? {
        assert(!Thread.isMainThread)

        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(self, options: nil, imageOut: &cgImage).assertNoError()
        return cgImage
    }

    func crop(to rect: CGRect) -> CVPixelBuffer? {
        assert(!Thread.isMainThread)

        CVPixelBufferLockBaseAddress(self, .readOnly).assertReturnSuccess()
        defer {
            CVPixelBufferUnlockBaseAddress(self, .readOnly).assertReturnSuccess()
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(self) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)

        let imageChannels = 4 // ARGB
        let cropStartOffset = Int(rect.origin.y) * bytesPerRow + imageChannels * Int(rect.origin.x)
        let outBytesPerRow = Int(rect.width) * imageChannels

        var inBuffer = vImage_Buffer()
        inBuffer.height = vImagePixelCount(rect.height)
        inBuffer.width = vImagePixelCount(rect.width)
        inBuffer.rowBytes = bytesPerRow

        inBuffer.data = baseAddress + cropStartOffset

        guard let croppedImageBytes = malloc(Int(rect.height) * outBytesPerRow) else {
            return nil
        }

        var outBuffer = vImage_Buffer(
            data: croppedImageBytes,
            height: vImagePixelCount(rect.height),
            width: vImagePixelCount(rect.width),
            rowBytes: outBytesPerRow
        )

        guard vImageScale_ARGB8888(&inBuffer, &outBuffer, nil, 0) == kvImageNoError else {
            free(croppedImageBytes)
            return nil
        }

        return croppedImageBytes.toCVPixelBuffer(
            pixelBuffer: self,
            targetWidth: Int(rect.width),
            targetHeight: Int(rect.height),
            targetImageRowBytes: outBytesPerRow
        )
    }

    func flip() -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }

        let width = UInt(CVPixelBufferGetWidth(self))
        let height = UInt(CVPixelBufferGetHeight(self))
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
        let outputImageRowBytes = inputImageRowBytes

        var inBuffer = vImage_Buffer(
            data: baseAddress,
            height: height,
            width: width,
            rowBytes: inputImageRowBytes)

        guard let targetImageBytes = malloc(Int(height) * outputImageRowBytes) else {
            return nil
        }
        var outBuffer = vImage_Buffer(data: targetImageBytes, height: height, width: width, rowBytes: outputImageRowBytes)

        // See https://developer.apple.com/documentation/accelerate/vimage/vimage_operations/image_reflection for other transformations
        let reflectError = vImageHorizontalReflect_ARGB8888(&inBuffer, &outBuffer, vImage_Flags(0))
        // let reflectError = vImageVerticalReflect_ARGB8888(&inBuffer, &outBuffer, vImage_Flags(0))

        guard reflectError == kvImageNoError else {
            free(targetImageBytes)
            return nil
        }

        return targetImageBytes.toCVPixelBuffer(pixelBuffer: self, targetWidth: Int(width), targetHeight: Int(height), targetImageRowBytes: outputImageRowBytes)
    }
}

extension UnsafeMutableRawPointer {
    // Converts the vImage buffer to CVPixelBuffer
    func toCVPixelBuffer(pixelBuffer: CVPixelBuffer, targetWidth: Int, targetHeight: Int, targetImageRowBytes: Int) -> CVPixelBuffer? {
        let pixelBufferType = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }

        var targetPixelBuffer: CVPixelBuffer?
        let conversionStatus = CVPixelBufferCreateWithBytes(nil, targetWidth, targetHeight, pixelBufferType, self, targetImageRowBytes, releaseCallBack, nil, nil, &targetPixelBuffer)

        guard conversionStatus == kCVReturnSuccess else {
            free(self)
            return nil
        }

        return targetPixelBuffer
    }
}

extension OSStatus {
    func assertNoError() {
        assert(self == 0)
    }
}

extension CVReturn {
    func assertReturnSuccess() {
        assert(self == kCVReturnSuccess)
    }
}
