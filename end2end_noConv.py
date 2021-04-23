from finn.util.visualization import showSrc, showInNetron
from finn.util.basic import make_build_dir
import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fold_constants import FoldConstants
from finn.util.pytorch import ToTensor
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.core.datatype import DataType
from finn.transformation.insert_topk import InsertTopK
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import RemoveUnusedTensors
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.custom_op.registry import getCustomOp
from finn.util.basic import pynq_part_map
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild


# define build dir
build_dir = "/workspace/finn"


# download test model, export to onnx
tfc = get_test_model_trained("TFC", 1, 1)
bo.export_finn_onnx(tfc, (1, 1, 28, 28), build_dir+"/tfc_w1_a1.onnx")


# do FINN wrap on onnx model
model = ModelWrapper(build_dir+"/tfc_w1_a1.onnx")


# do cleanup transformations that should almost always be done
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())


# save processed model
model.save(build_dir+"/tfc_w1_a1_tidy.onnx")


# add preprocessing
model = ModelWrapper(build_dir+"/tfc_w1_a1_tidy.onnx")
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)
## preprocessing: torchvision's ToTensor divides uint8 inputs by 255
totensor_pyt = ToTensor()
chkpt_preproc_name = build_dir+"/tfc_w1_a1_preproc.onnx"
bo.export_finn_onnx(totensor_pyt, ishape, chkpt_preproc_name)
## join preprocessing and core model
pre_model = ModelWrapper(chkpt_preproc_name)
model = model.transform(MergeONNXModels(pre_model))
## add input quantization annotation: UINT8 for all BNN-PYNQ models
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType.UINT8)
## save new model
model.save(build_dir+"/tfc_w1_a1_with_preproc.onnx")


# add postprocessing
## postprocessing: insert Top-1 node at the end
model = model.transform(InsertTopK(k=1))
chkpt_name = build_dir+"/tfc_w1_a1_pre_post.onnx"
## tidy-up again
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(chkpt_name)


# do streamlining (MAY NOT WORK FOR MANY NETWORKS, DISABLE WHEN THERE ARE PROBLEMS)
model = ModelWrapper(build_dir+"/tfc_w1_a1_pre_post.onnx")
## move initial Mul (from preproc) past the Reshape
model = model.transform(MoveScalarLinearPastInvariants())
## streamline
model = model.transform(Streamline())
model.save(build_dir+"/tfc_w1_a1_streamlined.onnx")


#transform bipolar matrix multiplications into xnorpopcount operations (important prerequisite for HLS)
model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
## absorb final add-mul nodes into TopK
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(RoundAndClipThresholds())
## bit of tidy-up
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save(build_dir+"/tfc_w1a1_ready_for_hls_conversion.onnx")

#convert xnorpopcountmatmul to corresponding functions in finn-hls-library
model = ModelWrapper(build_dir+"/tfc_w1a1_ready_for_hls_conversion.onnx")
model = model.transform(to_hls.InferBinaryStreamingFCLayer("decoupled"))
## TopK to LabelSelect
model = model.transform(to_hls.InferLabelSelectLayer())
## input quantization (if any) to standalone thresholding
model = model.transform(to_hls.InferThresholdingLayer())
model.save(build_dir+"/tfc_w1_a1_hls_layers.onnx")

# single out HLS layers, replace them with placeholder layers to signle out remaining non-hls(onnx)-layers
model = ModelWrapper(build_dir+"/tfc_w1_a1_hls_layers.onnx")
parent_model = model.transform(CreateDataflowPartition())
parent_model.save(build_dir+"/tfc_w1_a1_dataflow_parent.onnx")

# don't know what this is for, extracted hls layers (StreamingFC) have been moved to child dataflow model
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")

# load child model
model = ModelWrapper(dataflow_model_filename)

# adjust folding parameters PE and SIMD, the higher both are, the faster the generated accelerator will run, the more fpga resources will be consumed
fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
## (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
## ramstyle specifies how weights are stored (BRAM, LUTRAM...), 'auto' to let Vivado decide
## inFIFODepth and outFIFODepth specify FIFO depths needed by node from sorroungding FIFOs \/-Ü-\/
## all of this seems to be done automatically through ZynqBuild or VitisBuild transformations
config = [
    (16, 49, 16, 64, "block"),
    (8, 8, 64, 64, "auto"),
    (8, 8, 64, 64, "auto"),
    (10, 8, 64, 10, "distributed"),
]
for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
    fcl_inst = getCustomOp(fcl)
    fcl_inst.set_nodeattr("PE", pe)
    fcl_inst.set_nodeattr("SIMD", simd)
    fcl_inst.set_nodeattr("inFIFODepth", ififo)
    fcl_inst.set_nodeattr("outFIFODepth", ofifo)
    fcl_inst.set_nodeattr("ram_style", ramstyle)
## set parallelism for input quantizer to be same as first layer's SIMD
inp_qnt_node = model.get_nodes_by_op_type("Thresholding_Batch")[0]
inp_qnt = getCustomOp(inp_qnt_node)
inp_qnt.set_nodeattr("PE", 49)

# save model
model.save(build_dir+"/tfc_w1_a1_set_folding_factors.onnx")

# print names of supported PYNQ boards 
print(pynq_part_map.keys())

# change this if you have a different PYNQ board, see list above
pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board]
target_clk_ns = 10

# generate HLS code
model = ModelWrapper(build_dir+"/tfc_w1_a1_set_folding_factors.onnx")
model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns))
model.save(build_dir + "/tfc_w1_a1_post_synthesis.onnx")