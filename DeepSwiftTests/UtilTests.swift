//
//  UtilTests.swift
//  ConvNetSwift
//
//  Created by Alex on 11/24/15.
//  Copyright © 2015 OWL. All rights reserved.
//

import XCTest

class UtilTests: XCTestCase {

    func testImageConversion() {
        
        guard let image = UIImage(named: "Nyura.png") else {
            print("error: no image found on provided path")
            XCTAssert(false)
            return
        }
        
        let vol = image.toVol()!
        let newImage = vol.toImage()
        XCTAssertNotNil(newImage)
        
        let newVol = newImage!.toVol()!
        
        XCTAssertEqual(vol.w.count, newVol.w.count)
        
        for i: Int in 0 ..< vol.w.count {
            let equals = vol.w[i] == newVol.w[i]
            XCTAssert(equals)
            if !equals {
                print(vol.w[i], i)
                print(newVol.w[i], i)
                break
            }
        }
    }
    
    func testImgToArrayAndBackAgain() {
        guard let img = UIImage(named: "Nyura.png") else {
            print("error: no image found on provided path")
            XCTAssert(false)
            return
        }
        
        guard let image = img.cgImage else {
            print("error: no image found")
            XCTAssert(false)
            return
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
        
        let dataCtxt = context?.data
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(dataCtxt)!, count: width*height*components, deallocator: .free)
        
        var pixelMem = [UInt8](repeating: 0, count: data.count)
        (data as NSData).getBytes(&pixelMem, length: data.count)
        
        let doubleNormArray = pixelMem.map { (elem: UInt8) -> Double in
            return Double(elem)/255.0
        }
        
        let vol = Vol(width: width, height: height, depth: components, array: doubleNormArray)
        
        //=========================//
        
        for i in 0..<vol.sx {
            for j in 0..<vol.sy{
                if i<vol.sx/2 && j<vol.sy/2 {
                    vol.set(x: i, y: j, d: 0, v: 0.75)
                }
                if i>vol.sx/2 && j<vol.sy/2 {
                    vol.set(x: i, y: j, d: 1, v: 0.75)
                }
                if i<vol.sx/2 && j>vol.sy/2 {
                    vol.set(x: i, y: j, d: 2, v: 0.75)
                }
                if i>vol.sx/2 && j>vol.sy/2 {
                    vol.set(x: i, y: j, d: 3, v: 0.5)
                }
            }
        }
        
        let intDenormArray: [UInt8] = vol.w.map { (elem: Double) -> UInt8 in
            return UInt8(elem * 255.0)
        }
        
        let bitsPerPixel = bitsPerComponent * components
        
        let providerRef = CGDataProvider(
            data: Data(bytes: UnsafePointer<UInt8>(intDenormArray), count: intDenormArray.count * components) as CFData
        )
        
        let cgim = CGImage(
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
        )
        
        let newImage = UIImage(cgImage: cgim!)
        print(newImage)
    }
}
