
import Foundation

// Volume utilities
// intended for use with data augmentation
// crop is the size of output
// dx,dy are offset wrt incoming volume, of the shift
// fliplr is boolean on whether we also want to flip left<->right
func augment(V: Vol, crop:Int, dx: Int?, dy: Int?, fliplr: Bool = false) -> Vol {
    // note assumes square outputs of size crop x crop
    let dx = dx ?? RandUtils.randi(0, V.sx - crop)
    let dy = dy ?? RandUtils.randi(0, V.sy - crop)
    
    // randomly sample a crop in the input volume
    var W: Vol
    if(crop != V.sx || dx != 0 || dy != 0) {
        W = Vol(crop, crop, V.depth, 0.0)
        for x in 0 ..< crop {

            for y in 0 ..< crop {

                if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) {
                    continue // oob
                }
                for d in 0 ..< V.depth {

                    W.set(x,y,d,V.get(x+dx,y+dy,d)) // copy data over
                }
            }
        }
    } else {
        W = V
    }
    
    if(fliplr) {
        // flip volume horizontally
        let W2 = W.cloneAndZero()
        for x in 0 ..< W.sx {

            for y in 0 ..< W.sy {

                for d in 0 ..< W.depth {

                    W2.set(x,y,d,W.get(W.sx - x - 1,y,d)) // copy data over
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
//    if(convert_grayscale) {
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

public struct PixelData {
    var r:UInt8 = 255
    var g:UInt8
    var b:UInt8
    var a:UInt8
}

//private let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
//private let bitmapInfo:CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedFirst.rawValue)

public func imageFromARGB32Bitmap(pixels:[PixelData], width:UInt, height:UInt) -> UIImage {
    let bitsPerComponent:UInt = 8
    let bitsPerPixel:UInt = bitsPerComponent * 4
    
    assert(pixels.count == Int(width * height))
    
    var data = pixels // Copy to mutable []
    let providerRef = CGDataProviderCreateWithCFData(
        NSData(bytes: &data, length: data.count * sizeof(PixelData))
    )
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedFirst.rawValue)

    let cgim = CGImageCreate(
        Int(width),
        Int(height),
        Int(bitsPerComponent),
        Int(bitsPerPixel),
        Int(width * UInt(sizeof(PixelData))),
        rgbColorSpace,
        bitmapInfo,
        providerRef,
        nil,
        true,
        .RenderingIntentDefault
    )
    return UIImage(CGImage: cgim!)
}

extension Vol {
    func toImage() -> UIImage {
        var pixel_data: [PixelData] = []
        for i in 0 ..< self.sx {
            for j in 0 ..< self.sy {
                let r = UInt8(Int((get(i, j, 1))*255))
                let g = UInt8(Int((get(i, j, 2))*255))
                let b = UInt8(Int((get(i, j, 3))*255))
                let a = UInt8(Int((get(i, j, 0))*255))
                
                pixel_data.append(PixelData(r: r, g: g, b: b, a: a))
            }
        }
        
        return imageFromARGB32Bitmap(pixel_data, width: UInt(self.sx), height: UInt(self.sy))
    }
}

extension UIImage {
    func toVol(convert_grayscale convert_grayscale: Bool = false) -> Vol {
        
        if convert_grayscale {
            
        }
        
        let image = self.CGImage
        let width = CGImageGetWidth(image)
        let height = CGImageGetHeight(image)
        let bytesPerRow = (4 * width)
        let bitsPerComponent: Int = 8
//        let bitmapByteCount = bytesPerRow * height
        let pixels = calloc(height * width, sizeof(UInt32))//malloc(bitmapByteCount)
        
//        var bitmapInfo: CGBitmapInfo = [.ByteOrder32Big]
//        bitmapInfo |= CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedLast.rawValue)
        let bitmapInfo:CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedFirst.rawValue | CGBitmapInfo.ByteOrder32Big.rawValue)
        //  ~CGBitmapInfo.AlphaInfoMask.rawValue |
        let colorSpace = CGColorSpaceCreateDeviceRGB()//CGImageGetColorSpace(image)
        
        let context = CGBitmapContextCreate(
            pixels,
            width,
            height,
            bitsPerComponent,
            bytesPerRow,
            colorSpace,
            bitmapInfo.rawValue)
        
        CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), image)

        let dataCtxt = CGBitmapContextGetData(context)
//         let data = UnsafePointer<UInt8>(dataCtxt)
        let data = NSData(bytesNoCopy: dataCtxt, length: width*height*4, freeWhenDone: true)
        
        var pixelMem = [UInt8](count:data.length, repeatedValue:0)
        data.getBytes(&pixelMem, length:data.length)
        
        let vol = Vol(width, height, 4)

        for x in 0 ..< width {
            for y in 0 ..< height {
                //Here is your raw pixels
                let offset = 4*((Int(width) * Int(y)) + Int(x))
                
                let alpha = normalize(pixelMem[offset])

                let red = normalize(pixelMem[(offset+1)])
                let green = normalize(pixelMem[(offset+2)])
                let blue = normalize(pixelMem[(offset+3)])

                vol.set(x, y, 0, red)
                vol.set(x, y, 1, green)
                vol.set(x, y, 2, blue)
                vol.set(x, y, 3, alpha)
            }
        }

        return vol
    }
    
    private func normalize(pixelChannel: UInt8) -> Double{
//        let unsafeUINT8 = UnsafePointer<UInt8>(nilLiteral: pixelChannel)
        return Double(pixelChannel)/255.0
    }
}

