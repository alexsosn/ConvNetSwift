
protocol LayerOptTypeProtocol {
    var layerType: LayerType {get}
}

protocol LayerOutOptProtocol: LayerOptTypeProtocol {
    var outSx: Int {get set}
    var outSy: Int {get set}
    var outDepth: Int {get set}
}

protocol LayerInOptProtocol: LayerOptTypeProtocol {
    var inSx: Int {get set}
    var inSy: Int {get set}
    var inDepth: Int {get set}
}

protocol LayerOptActivationProtocol {
    var activation: ActivationType {get set}
}

protocol DropProbProtocol {
    var dropProb: Double? {get set}
}
