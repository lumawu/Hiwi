import onnx
from finn.util.visualization import showSrc, showInNetron
from brevitas_examples import bnn_pynq
from finn.util.test import get_test_model
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper

print("############showsrc############")
showSrc(bnn_pynq.models.FC)

print("############downloadmodel############")
lfc = get_test_model(netname = "LFC", wbits = 1, abits = 1, pretrained = True)

print("############showmodel############")
lfc

print("############exportmodelonnx############")
export_onnx_path = "/tmp/LFCW1A1.onnx"
input_shape = (1, 1, 28, 28)
bo.export_finn_onnx(lfc, input_shape, export_onnx_path)

print("############exportmodelfinn############")
model = ModelWrapper(export_onnx_path)
model.graph.node[9]