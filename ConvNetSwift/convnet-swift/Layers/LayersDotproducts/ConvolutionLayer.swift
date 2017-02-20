//
//  ConvolutionLayer.swift
//  ConvNetSwift
//
//  Created by Alex on 2/17/17.
//  Copyright © 2017 OWL. All rights reserved.
//


// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar
import Foundation

struct ConvLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    var layerType: LayerType = .Conv
    
    var filters: Int
    var sx: Int
    var sy: Int?
    var inDepth: Int = 0
    var inSx: Int = 0
    var inSy: Int = 0
    var stride: Int = 1
    var pad: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var biasPref: Double = 0.0
    var activation: ActivationType = .Undefined
    
    init (sx: Int, filters: Int, stride: Int, pad: Int, activation: ActivationType) {
        self.sx = sx
        self.filters = filters
        self.stride = stride
        self.pad = pad
        self.activation = activation
    }
    
}

class ConvLayer: InnerLayer {
    var outDepth: Int
    var sx: Int
    var sy: Int
    var inDepth: Int
    var inSx: Int
    var inSy: Int
    var stride: Int = 1
    var pad: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var filters: [Vol]
    var biases: Vol
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: ConvLayerOpt) {
        
        // required
        outDepth = opt.filters
        sx = opt.sx // filter size. Should be odd if possible, it's cleaner.
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        sy = opt.sy ?? opt.sx
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        outSx = Int(floor(Double(inSx + pad * 2 - sx) / Double(stride + 1)))
        outSy = Int(floor(Double(inSy + pad * 2 - sy) / Double(stride + 1)))
        layerType = .Conv
        
        // initializations
        let bias = opt.biasPref
        filters = []
        for _ in 0..<outDepth {
            filters.append(Vol(sx: sx, sy: sy, depth: inDepth))
        }
        biases = Vol(sx: 1, sy: 1, depth: outDepth, c: bias)
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        inAct = V
        let A = Vol(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        
        let V_sx = V.sx
        let V_sy = V.sy
        let xy_stride = stride
        
        for d in 0 ..< outDepth {
            let f = filters[d]
            var x = -pad
            var y = -pad
            
            for ay in 0 ..< outSy {
                y+=xy_stride // xy_stride
                x = -pad
                
                for ax in 0 ..< outSx {  // xy_stride
                    x+=xy_stride
                    // convolve centered at this particular location
                    var a: Double = 0.0
                    
                    for fy in 0 ..< f.sy {
                        let oy = y+fy // coordinates in the original input array coordinates
                        
                        for fx in 0 ..< f.sx {
                            let ox = x+fx
                            if oy>=0 && oy<V_sy && ox>=0 && ox<V_sx {
                                
                                for fd in 0 ..< f.depth {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd]
                                }
                            }
                        }
                    }
                    a += biases.w[d]
                    A.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        outAct = A
        return outAct!
    }
    
    func backward() -> () {
        
        guard
            let V = inAct,
            let outAct = outAct
            else {
                return
        }
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out gradient wrt bottom data, we're about to fill it
        
        let V_sx = V.sx
        let V_sy = V.sy
        let xy_stride = stride
        
        for d in 0 ..< outDepth {
            
            let f = filters[d]
            var x = -pad
            var y = -pad
            for ay in 0 ..< outSy {  // xy_stride
                y+=xy_stride
                x = -pad
                for ax in 0 ..< outSx {  // xy_stride
                    x+=xy_stride
                    
                    // convolve centered at this particular location
                    let chainGrad = outAct.getGrad(x: ax, y: ay, d: d) // gradient from above, from chain rule
                    for fy in 0 ..< f.sy {
                        
                        let oy = y+fy // coordinates in the original input array coordinates
                        for fx in 0 ..< f.sx {
                            
                            let ox = x+fx
                            if oy>=0 && oy<V_sy && ox>=0 && ox<V_sx {
                                for fd in 0 ..< f.depth {
                                    
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    let ix1 = ((V_sx * oy)+ox)*V.depth+fd
                                    let ix2 = ((f.sx * fy)+fx)*f.depth+fd
                                    f.dw[ix2] += V.w[ix1]*chainGrad
                                    V.dw[ix1] += f.w[ix2]*chainGrad
                                }
                            }
                        }
                    }
                    biases.dw[d] += chainGrad
                }
            }
            filters[d] = f
        }
        //        inAct = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< outDepth {
            
            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1DecayMul: l1DecayMul,
                l2DecayMul: l2DecayMul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1DecayMul: 0.0,
            l2DecayMul: 0.0))
        return response
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        assert(filters.count + 1 == paramsAndGrads.count)
        
        for i in 0 ..< outDepth {
            filters[i].w = paramsAndGrads[i].params
            filters[i].dw = paramsAndGrads[i].grads
        }
        biases.w = paramsAndGrads.last!.params
        biases.dw = paramsAndGrads.last!.grads
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["sx"] = sx as AnyObject? // filter size in x, y dims
        json["sy"] = sy as AnyObject?
        json["stride"] = stride as AnyObject?
        json["inDepth"] = inDepth as AnyObject?
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["l1DecayMul"] = l1DecayMul as AnyObject?
        json["l2DecayMul"] = l2DecayMul as AnyObject?
        json["pad"] = pad as AnyObject?
        
        var jsonFilters: [[String: AnyObject]] = []
        for i in 0 ..< filters.count {
            jsonFilters.append(filters[i].toJSON())
        }
        json["filters"] = jsonFilters as AnyObject?
        
        json["biases"] = biases.toJSON() as AnyObject?
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        outDepth = json["outDepth"] as! Int
    //        outSx = json["outSx"] as! Int
    //        outSy = json["outSy"] as! Int
    //        layerType = json["layerType"] as! String
    //        sx = json["sx"] as! Int // filter size in x, y dims
    //        sy = json["sy"] as! Int
    //        stride = json["stride"] as! Int
    //        inDepth = json["inDepth"] as! Int // depth of input volume
    //        filters = []
    ////        l1DecayMul = json["l1DecayMul"] != nil ? json["l1DecayMul"] : 1.0
    ////        l2DecayMul = json["l2DecayMul"] != nil ? json["l2DecayMul"] : 1.0
    ////        pad = json["pad"] != nil ? json["pad"] : 0
    //
    //        var jsonFilters = json["filters"] as! [[String: AnyObject]]
    //        for i in 0 ..< jsonFilters.count {
    //            let v = Vol(0,0,0,0)
    //            v.fromJSON(jsonFilters[i])
    //            filters.append(v)
    //        }
    //        
    //        biases = Vol(0,0,0,0)
    //        biases.fromJSON(json["biases"] as! [String: AnyObject])
    //    }
}
