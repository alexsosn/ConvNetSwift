
protocol LayerOptTypeProtocol {
    var layer_type: LayerType {get}
}

protocol LayerOutOptProtocol: LayerOptTypeProtocol {
    var out_sx: Int {get set}
    var out_sy: Int {get set}
    var out_depth: Int {get set}
}

protocol LayerInOptProtocol: LayerOptTypeProtocol {
    var in_sx: Int {get set}
    var in_sy: Int {get set}
    var in_depth: Int {get set}

}

protocol LayerOptActivationProtocol {
    var activation: ActivationType {get set}
}

protocol DropProbProtocol {
    var drop_prob: Double? {get set}
}