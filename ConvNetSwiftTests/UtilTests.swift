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
        
        let vol = image.toVol()
        let newImage = vol.toImage()
        XCTAssertNotNil(newImage)
        
        let newVol = newImage!.toVol()
        
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
        
        let image = img.CGImage
        let width = CGImageGetWidth(image)
        let height = CGImageGetHeight(image)
        let components = 4
        let bytesPerRow = (components * width)
        let bitsPerComponent: Int = 8
        let pixels = calloc(height * width, sizeof(UInt32))
        
        let bitmapInfo: CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedLast.rawValue | CGBitmapInfo.ByteOrder32Big.rawValue)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
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
        let data = NSData(bytesNoCopy: dataCtxt, length: width*height*components, freeWhenDone: true)
        
        var pixelMem = [UInt8](count: data.length, repeatedValue: 0)
        data.getBytes(&pixelMem, length: data.length)
        
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
        
        let providerRef = CGDataProviderCreateWithCFData(
            NSData(bytes: intDenormArray, length: intDenormArray.count * components)
        )
        
        let cgim = CGImageCreate(
            width,
            height,
            bitsPerComponent,
            bitsPerPixel,
            bytesPerRow,
            colorSpace,
            bitmapInfo,
            providerRef,
            nil,
            false,
            .RenderingIntentDefault
        )
        
        let newImage = UIImage(CGImage: cgim!)
        print(newImage)
    }
}
