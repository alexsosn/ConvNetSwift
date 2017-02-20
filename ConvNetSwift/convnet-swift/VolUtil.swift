
import Foundation

// Volume utilities
// intended for use with data augmentation
// crop is the size of output
// dx,dy are offset wrt incoming volume, of the shift
// fliplr is boolean on whether we also want to flip left<->right
func augment(_ V: Vol, crop: Int, dx: Int?, dy: Int?, fliplr: Bool = false) -> Vol {
    // note assumes square outputs of size crop x crop
    let dx = dx ?? RandUtils.randi(0, V.sx - crop)
    let dy = dy ?? RandUtils.randi(0, V.sy - crop)
    
    // randomly sample a crop in the input volume
    var W: Vol
    if crop != V.sx || dx != 0 || dy != 0 {
        W = Vol(sx: crop, sy: crop, depth: V.depth, c: 0.0)
        for x in 0 ..< crop {

            for y in 0 ..< crop {

                if x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy {
                    continue // oob
                }
                for d in 0 ..< V.depth {

                    W.set(x: x, y: y, d: d, v: V.get(x: x+dx, y: y+dy, d: d)) // copy data over
                }
            }
        }
    } else {  
        W = V
    }
    
    if fliplr {
        // flip volume horizontally
        let W2 = W.cloneAndZero()
        for x in 0 ..< W.sx {

            for y in 0 ..< W.sy {

                for d in 0 ..< W.depth {

                    W2.set(x: x, y: y, d: d, v: W.get(x: W.sx - x - 1, y: y, d: d)) // copy data over
                }
            }
        }
        W = W2 //swap
    }
    return W
}

import UIKit
import CoreGraphics

// img is a DOM element that contains a loaded image
// returns a Vol of size (W, H, 4). 4 is for RGBA
//func img_to_vol(img: UIImage, convert_grayscale: Bool = false) -> Vol? {
//    
////    guard let uiimage = UIImage(contentsOfFile: "/PATH/TO/image.png") else {
////        print("error: no image found on provided path")
////        return nil
////    }
//    var image = img.CGImage
//    
//    
//    let width = CGImageGetWidth(image)
//    let height = CGImageGetHeight(image)
//    let colorspace = CGColorSpaceCreateDeviceRGB()
//    let bytesPerRow = (4 * width)
//    let bitsPerComponent: Int = 8
//    var pixels = UnsafeMutablePointer<Void>(malloc(width*height*4))
//    
//    
//    var context = CGBitmapContextCreate(
//        pixels,
//        width,
//        height,
//        bitsPerComponent,
//        bytesPerRow,
//        colorspace,
//        CGBitmapInfo().rawValue)
//    
//    CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), image)
//    
//    
//    for x in 0...width {
//        for y in 0...height {
//            //Here is your raw pixels
//            let offset = 4*((Int(width) * Int(y)) + Int(x))
//            let alpha = pixels[offset]
//            let red = pixels[offset+1]
//            let green = pixels[offset+2]
//            let blue = pixels[offset+3]
//        }
//    }
//    ////////////////////////////
//    // prepare the input: get pixels and normalize them
//    var pv = []
//    for i in 0 ..< width*height {
//
//        pv.append(p[i]/255.0-0.5) // normalize image pixels to [-0.5, 0.5]
//    }
//    
//    var x = Vol(W, H, 4, 0.0) //input volume (image)
//    x.w = pv
//    
//    if convert_grayscale {
//        // flatten into depth=1 array
//        var x1 = Vol(width, height, 1, 0.0)
//        for i in 0 ..< width {
//
//            for j in 0 ..< height {
//
//                x1.set(i, j, 0, x.get(i,j,0))
//            }
//        }
//        x = x1
//    }
//    
//    return x
//}


extension Vol {
    
    func denormalize(_ pixelChannel: Double) -> UInt8{
        return UInt8(pixelChannel * 255.0)
    }
    
    func toImage() -> UIImage? {
        
        let intDenormArray: [UInt8] = w.map { (elem: Double) -> UInt8 in
            return denormalize(elem)
        }
        
        let width = sx
        let height = sy
        let components = depth
        let bitsPerComponent: Int = 8
        let bitsPerPixel = bitsPerComponent * components
        let bytesPerRow = (components * width)
        let bitmapInfo: CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let providerRef = CGDataProvider(
            data: Data(bytes: UnsafePointer<UInt8>(intDenormArray), count: intDenormArray.count * components) as CFData
        )
        
        guard let cgim = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: providerRef!,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
            )  else {
                return nil
        }
        
        let newImage = UIImage(cgImage: cgim)
        return newImage
    }
}

extension UIImage {
    func toVol(convert_grayscale: Bool = false) -> Vol? {
        
        if convert_grayscale {
            // TODO: implement
        }
        
        guard let image = cgImage else {
            return nil
        }
        
        let width = image.width
        let height = image.height
        let components = 4
        let bytesPerRow = (components * width)
        let bitsPerComponent: Int = 8
        let pixels = calloc(height * width, MemoryLayout<UInt32>.size)
        
        let bitmapInfo: CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        let context = CGContext(
            data: pixels,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue)
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        let dataCtxt = context?.data!
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(dataCtxt)!, count: width*height*components, deallocator: .free)
        
        var pixelMem = [UInt8](repeating: 0, count: data.count)
        (data as NSData).getBytes(&pixelMem, length: data.count)
        
        let doubleNormArray = pixelMem.map { (elem: UInt8) -> Double in
            return normalize(elem)
        }
        
        let vol = Vol(width: width, height: height, depth: components, array: doubleNormArray)
        
        return vol
    }
    
    fileprivate func normalize(_ pixelChannel: UInt8) -> Double{
        return Double(pixelChannel)/255.0
    }
}

