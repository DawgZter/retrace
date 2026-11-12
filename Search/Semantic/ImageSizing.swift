import CoreGraphics
import Foundation

/// How a screenshot is sized for EmbeddingGemma 2's image encoder.
enum ImageSizing {
    /// The size that fills `budget` soft tokens while keeping the aspect ratio: each token is 3 × 3 patches of
    /// 16 px, and each side is rounded down to a multiple of 48 px. This is the reference image processor's rule,
    /// as ported in DigUp's `ImageSizing` (https://github.com/ARahim3/DigUp, MIT, © Abdur Rahim). Nil for an
    /// image with no area.
    static func targetSize(width: Int, height: Int, budget: Int) -> (width: Int, height: Int)? {
        guard height > 0, width > 0, budget > 0 else { return nil }
        let patch = 16, pooling = 3
        let maxPatches = budget * pooling * pooling
        let factor = (Double(maxPatches * patch * patch) / Double(height * width)).squareRoot()
        let side = pooling * patch
        var targetHeight = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var targetWidth = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if targetHeight == 0 && targetWidth == 0 { return nil }
        let maxSide = (maxPatches / (pooling * pooling)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(width / height * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(height / width * side, maxSide)
        }
        return (targetWidth, targetHeight)
    }

    /// `image` scaled to its target size for `budget`, as packed RGB. Nil when it can't be drawn.
    static func embeddingImage(from image: CGImage, budget: Int) -> EmbeddingImage? {
        guard let size = targetSize(width: image.width, height: image.height, budget: budget) else { return nil }
        let bytesPerRow = size.width * 4
        var rgba = [UInt8](repeating: 0, count: bytesPerRow * size.height)
        let drawn: Bool = rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: size.width,
                height: size.height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
            return true
        }
        guard drawn else { return nil }

        var rgb = Data(count: size.width * size.height * 3)
        rgb.withUnsafeMutableBytes { destination in
            let out = destination.bindMemory(to: UInt8.self)
            var read = 0, write = 0
            while read < rgba.count {
                out[write] = rgba[read]
                out[write + 1] = rgba[read + 1]
                out[write + 2] = rgba[read + 2]
                read += 4
                write += 3
            }
        }
        return EmbeddingImage(width: size.width, height: size.height, rgb: rgb)
    }
}
