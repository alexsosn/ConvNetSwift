import Foundation

struct PoolLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Pool

    var sx: Int
    var sy: Int = 0
    var inDepth: Int = 0
    var inSx: Int = 0
    var inSy: Int = 0
    var stride: Int? = nil
    var pad: Int? = nil
    
    init (sx: Int, stride: Int) {
        self.sx = sx
        self.stride = stride
    }
}

class PoolLayer: InnerLayer {
    var sx: Int
    var sy: Int
    var inDepth: Int
    var inSx: Int
    var inSy: Int
    var stride: Int = 2
    var pad: Int = 0
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var switchx: [Int]
    var switchy: [Int]
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: PoolLayerOpt){
        
        // required
        sx = opt.sx // filter size
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        sy = opt.sy
        stride = opt.stride ?? 2
        pad = opt.pad ?? 0 // amount of 0 padding to add around borders of input volume
        
        // computed
        outDepth = inDepth
        outSx = (inSx + pad * 2 - sx) / stride + 1
        outSy = (inSy + pad * 2 - sy) / stride + 1
        layerType = .Pool
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        switchx = ArrayUtils.zerosInt(outSx*outSy*outDepth)
        switchy = ArrayUtils.zerosInt(outSx*outSy*outDepth)
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        
        let A = Vol(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        
        var n = 0 // a counter for switches
        for d in 0 ..< outDepth {

            var x = -pad
            var y = -pad
            for ax in 0 ..< outSx {
                y = -pad
                
                for ay in 0 ..< outSy {
                    
                    // convolve centered at this particular location
                    var a = -99999.0 // hopefully small enough ;\
                    var winx = -1, winy = -1
                    for fx in 0 ..< sx {

                        for fy in 0 ..< sy {

                            let oy = y+fy
                            let ox = x+fx
                            if oy>=0 && oy<V.sy && ox>=0 && ox<V.sx {
                                let v = V.get(x: ox, y: oy, d: d)
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if v > a {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }
                    switchx[n] = winx
                    switchy[n] = winy
                    n += 1
                    y+=stride
                    A.set(x: ax, y: ay, d: d, v: a)
                }

                x+=stride
            }
        }
        outAct = A
        return outAct!
    }
    
    func backward() -> () {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        guard let V = inAct else {
            fatalError("inAct is nil")
        }
        
        guard let outAct = outAct else {
            fatalError("outAct is nil")
        }
        
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out gradient wrt data
//        var A = outAct // computed in forward pass
        
        var n = 0
        for d in 0 ..< outDepth {
            
            var x = -pad
            for ax in 0 ..< outSx {
                
                var y = -pad
                for ay in 0 ..< outSy {
                    
                    let chainGrad = outAct.getGrad(x: ax, y: ay, d: d)
                    V.addGrad(x: switchx[n], y: switchy[n], d: d, v: chainGrad)
                    n += 1
                    y += stride
                }
                
                x += stride
            }
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["sx"] = sx as AnyObject?
        json["sy"] = sy as AnyObject?
        json["stride"] = stride as AnyObject?
        json["inDepth"] = inDepth as AnyObject?
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["pad"] = pad as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"]
//        outSx = json["outSx"]
//        outSy = json["outSy"]
//        layerType = json["layerType"]
//        sx = json["sx"]
//        sy = json["sy"]
//        stride = json["stride"]
//        inDepth = json["inDepth"]
//        pad = json["pad"] != nil ? json["pad"] : 0 // backwards compatibility
//        switchx = zeros(outSx*outSy*outDepth) // need to re-init these appropriately
//        switchy = zeros(outSx*outSy*outDepth)
//    }
}


