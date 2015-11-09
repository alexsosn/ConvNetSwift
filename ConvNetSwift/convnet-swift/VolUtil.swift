
import Foundation

// Volume utilities
// intended for use with data augmentation
// crop is the size of output
// dx,dy are offset wrt incoming volume, of the shift
// fliplr is boolean on whether we also want to flip left<->right
func augment(V: Vol, crop:Int, dx: Int?, dy: Int?, fliplr: Bool = false) -> Vol {
    // note assumes square outputs of size crop x crop
    let dx = dx ?? randi(0, V.sx - crop)
    let dy = dy ?? randi(0, V.sy - crop)
    
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
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
}

//func vol_to_img(){
//    var pixels = [PixelData]()
//    
//    let red = PixelData(a: 255, r: 255, g: 0, b: 0)
//    let green = PixelData(a: 255, r: 0, g: 255, b: 0)
//    let blue = PixelData(a: 255, r: 0, g: 0, b: 255)
//    
//    for i in 1...300 {
//        pixels.append(red)
//    }
//    for i in 1...300 {
//        pixels.append(green)
//    }
//    for i in 1...300 {
//        pixels.append(blue)
//    }
//    
//    let image = imageFromARGB32Bitmap(pixels, 30, 30)
//}
