       гK"	  А< l╓Abrain.Event:2+MP╢Я     HTп-	ВкН< l╓A"Т╡

global_step/Initializer/zerosConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
s
input_producer/ConstConst*
dtype0*+
value"B B../data/valid-data.csv*
_output_shapes
:
U
input_producer/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Z
input_producer/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 
Т
input_producer/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
Ъ
#input_producer/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
А
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*
	summarize*

T
2
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
У
input_producerFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
Щ
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/Identity*

timeout_ms         *
Tcomponents
2
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
Y
input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
К
'input_producer/fraction_of_32_full/tagsConst*
dtype0*3
value*B( B"input_producer/fraction_of_32_full*
_output_shapes
: 
С
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
y
TextLineReaderV2TextLineReaderV2*
	container *
shared_name *
skip_header_lines *
_output_shapes
: 
_
ReaderReadUpToV2/num_recordsConst*
dtype0	*
value
B	 RЇ*
_output_shapes
: 
Ш
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:         :         
Y
ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
z

ExpandDims
ExpandDimsReaderReadUpToV2:1ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
h
DecodeCSV/record_defaults_0Const*
dtype0*
valueB*    *
_output_shapes
:
h
DecodeCSV/record_defaults_1Const*
dtype0*
valueB*    *
_output_shapes
:
h
DecodeCSV/record_defaults_2Const*
dtype0*
valueB*    *
_output_shapes
:
d
DecodeCSV/record_defaults_3Const*
dtype0*
valueB
B *
_output_shapes
:
d
DecodeCSV/record_defaults_4Const*
dtype0*
valueB
B *
_output_shapes
:
d
DecodeCSV/record_defaults_5Const*
dtype0*
valueB
B *
_output_shapes
:
Е
	DecodeCSV	DecodeCSV
ExpandDimsDecodeCSV/record_defaults_0DecodeCSV/record_defaults_1DecodeCSV/record_defaults_2DecodeCSV/record_defaults_3DecodeCSV/record_defaults_4DecodeCSV/record_defaults_5*
OUT_TYPE

2*
field_delim,*Ж
_output_shapest
r:         :         :         :         :         :         
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
╢
batch/fifo_queueFIFOQueueV2*
capacityИ'*
_output_shapes
: **
shapes 
:::::*
component_types	
2*
	container *
shared_name 
║
batch/fifo_queue_EnqueueManyQueueEnqueueManyV2batch/fifo_queueDecodeCSV:3DecodeCSV:4DecodeCSV:5DecodeCSV:1DecodeCSV:2*

timeout_ms         *
Tcomponents	
2
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *╖Q9*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
|
 batch/fraction_of_5000_full/tagsConst*
dtype0*,
value#B! Bbatch/fraction_of_5000_full*
_output_shapes
: 
z
batch/fraction_of_5000_fullScalarSummary batch/fraction_of_5000_full/tags	batch/mul*
T0*
_output_shapes
: 
J
batch/nConst*
dtype0*
value
B :Ї*
_output_shapes
: 
ф
batchQueueDequeueUpToV2batch/fifo_queuebatch/n*

timeout_ms         *
component_types	
2*s
_output_shapesa
_:         :         :         :         :         
`
ConstConst*
dtype0*'
valueBBpositiveBnegative*
_output_shapes
:
V
string_to_index/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
]
string_to_index/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
]
string_to_index/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ц
string_to_index/rangeRangestring_to_index/range/startstring_to_index/Sizestring_to_index/range/delta*

Tidx0*
_output_shapes
:
j
string_to_index/ToInt64Caststring_to_index/range*

DstT0	*

SrcT0*
_output_shapes
:
и
string_to_index/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 
k
 string_to_index/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
╗
%string_to_index/hash_table/table_initInitializeTablestring_to_index/hash_tableConststring_to_index/ToInt64*-
_class#
!loc:@string_to_index/hash_table*

Tkey0*

Tval0	
┌
hash_table_LookupLookupTableFindstring_to_index/hash_tablebatch:2 string_to_index/hash_table/Const*	
Tin0*-
_class#
!loc:@string_to_index/hash_table*

Tout0	*'
_output_shapes
:         
Х
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ShapeShapebatch*
out_type0*
T0*
_output_shapes
:
▌
Odnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/CastCastPdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
Ф
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
э
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/NotEqualNotEqualbatchSdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╫
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/WhereWhereSdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/NotEqual*'
_output_shapes
:         
л
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
·
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ReshapeReshapebatchXdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
п
^dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
¤
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_sliceStridedSlicePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Where^dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_1`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
│
bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
│
bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Й
Zdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1StridedSlicePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Where`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stackbdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_1bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
ч
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/unstackUnpackOdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
ш
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/stackPackTdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
╡
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/MulMulZdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/stack*
T0	*'
_output_shapes
:         
к
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
╥
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/SumSumNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Mul`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum/reduction_indices*#
_output_shapes
:         *
T0	*
	keep_dims( *

Tidx0
н
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/AddAddXdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_sliceNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
█
Qdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/GatherGatherRdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ReshapeNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
а
Mdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ConstConst*
dtype0*
valueBBax01Bax02*
_output_shapes
:
О
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Х
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Х
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ў
Mdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/rangeRangeSdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/startLdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/SizeSdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/delta*

Tidx0*
_output_shapes
:
┌
Odnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ToInt64CastMdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
▐
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
г
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
■
]dnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/table_initInitializeTableV2Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableMdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ConstOdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ToInt64*

Tkey0*

Tval0	
Ю
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/hash_table_LookupLookupTableFindV2Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableQdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/GatherXdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
б
Vdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDense/default_valueConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Е
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDenseSparseToDensePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/WhereOdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/CastLdnn/input_from_feature_columns/input_layer/alpha_indicator/hash_table_LookupVdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0	*0
_output_shapes
:                  
Н
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
П
Jdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
К
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
Р
Kdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
С
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┘
Bdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hotOneHotHdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDenseHdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/depthKdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/on_valueLdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/off_value*
axis         *
T0*4
_output_shapes"
 :                  *
TI0	
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
к
>dnn/input_from_feature_columns/input_layer/alpha_indicator/SumSumBdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hotPdnn/input_from_feature_columns/input_layer/alpha_indicator/Sum/reduction_indices*'
_output_shapes
:         *
T0*
	keep_dims( *

Tidx0
╛
@dnn/input_from_feature_columns/input_layer/alpha_indicator/ShapeShape>dnn/input_from_feature_columns/input_layer/alpha_indicator/Sum*
out_type0*
T0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
а
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/alpha_indicator/ShapeNdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stackPdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
М
Jdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
а
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_sliceJdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:*

axis 
Ч
Bdnn/input_from_feature_columns/input_layer/alpha_indicator/ReshapeReshape>dnn/input_from_feature_columns/input_layer/alpha_indicator/SumHdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
Ц
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ShapeShapebatch:1*
out_type0*
T0*
_output_shapes
:
█
Ndnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/CastCastOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
У
Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
э
Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/NotEqualNotEqualbatch:1Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╒
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/WhereWhereRdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/NotEqual*'
_output_shapes
:         
к
Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
·
Qdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ReshapeReshapebatch:1Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
о
]dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
°
Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_sliceStridedSliceOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Where]dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_1_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Д
Ydnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1StridedSliceOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Where_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stackadnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_1adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
х
Qdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/unstackUnpackNdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
ц
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/stackPackSdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
▓
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/MulMulYdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/stack*
T0	*'
_output_shapes
:         
й
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
╧
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/SumSumMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Mul_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum/reduction_indices*#
_output_shapes
:         *
T0	*
	keep_dims( *

Tidx0
к
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/AddAddWdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_sliceMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
╪
Pdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/GatherGatherQdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ReshapeMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
Ю
Kdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ConstConst*
dtype0*
valueBBbx01Bbx02*
_output_shapes
:
М
Jdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
У
Qdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
У
Qdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ю
Kdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/rangeRangeQdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/startJdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/SizeQdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/delta*

Tidx0*
_output_shapes
:
╓
Mdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ToInt64CastKdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
▄
Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
б
Vdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Ў
[dnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/table_initInitializeTableV2Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tableKdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ConstMdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ToInt64*

Tkey0*

Tval0	
Ш
Kdnn/input_from_feature_columns/input_layer/beta_indicator/hash_table_LookupLookupTableFindV2Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tablePdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/GatherVdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
а
Udnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDense/default_valueConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
А
Gdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDenseSparseToDenseOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/WhereNdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/CastKdnn/input_from_feature_columns/input_layer/beta_indicator/hash_table_LookupUdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0	*0
_output_shapes
:                  
М
Gdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
О
Idnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Й
Gdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
П
Jdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Р
Kdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╘
Adnn/input_from_feature_columns/input_layer/beta_indicator/one_hotOneHotGdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDenseGdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/depthJdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/on_valueKdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/off_value*
axis         *
T0*4
_output_shapes"
 :                  *
TI0	
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
з
=dnn/input_from_feature_columns/input_layer/beta_indicator/SumSumAdnn/input_from_feature_columns/input_layer/beta_indicator/one_hotOdnn/input_from_feature_columns/input_layer/beta_indicator/Sum/reduction_indices*'
_output_shapes
:         *
T0*
	keep_dims( *

Tidx0
╝
?dnn/input_from_feature_columns/input_layer/beta_indicator/ShapeShape=dnn/input_from_feature_columns/input_layer/beta_indicator/Sum*
out_type0*
T0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/beta_indicator/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/beta_indicator/ShapeMdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stackOdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Л
Idnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/beta_indicator/strided_sliceIdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:*

axis 
Ф
Adnn/input_from_feature_columns/input_layer/beta_indicator/ReshapeReshape=dnn/input_from_feature_columns/input_layer/beta_indicator/SumGdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
y
2dnn/input_from_feature_columns/input_layer/x/ShapeShapebatch:3*
out_type0*
T0*
_output_shapes
:
К
@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┌
:dnn/input_from_feature_columns/input_layer/x/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/x/Shape@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackBdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
~
<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ў
:dnn/input_from_feature_columns/input_layer/x/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/x/strided_slice<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1*
N*
T0*
_output_shapes
:*

axis 
─
4dnn/input_from_feature_columns/input_layer/x/ReshapeReshapebatch:3:dnn/input_from_feature_columns/input_layer/x/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
y
2dnn/input_from_feature_columns/input_layer/y/ShapeShapebatch:4*
out_type0*
T0*
_output_shapes
:
К
@dnn/input_from_feature_columns/input_layer/y/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┌
:dnn/input_from_feature_columns/input_layer/y/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/y/Shape@dnn/input_from_feature_columns/input_layer/y/strided_slice/stackBdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
~
<dnn/input_from_feature_columns/input_layer/y/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ў
:dnn/input_from_feature_columns/input_layer/y/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/y/strided_slice<dnn/input_from_feature_columns/input_layer/y/Reshape/shape/1*
N*
T0*
_output_shapes
:*

axis 
─
4dnn/input_from_feature_columns/input_layer/y/ReshapeReshapebatch:4:dnn/input_from_feature_columns/input_layer/y/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
п
1dnn/input_from_feature_columns/input_layer/concatConcatV2Bdnn/input_from_feature_columns/input_layer/alpha_indicator/ReshapeAdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape4dnn/input_from_feature_columns/input_layer/x/Reshape4dnn/input_from_feature_columns/input_layer/y/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0
╟
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   А   *
_output_shapes
:
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *гоX╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *гоX>*
_output_shapes
: 
в
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes
:	А*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
▒
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
г
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
╦
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
Ш
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А
▓
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
┤
1dnn/hiddenlayer_0/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBА*    *
_output_shapes	
:А
┴
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
З
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes	
:А
л
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes	
:А
v
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes
:	А
╔
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
p
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes	
:А
в
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*(
_output_shapes
:         А*
T0*
data_formatNHWC
z
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:         А
[
dnn/zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
С
dnn/zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*(
_output_shapes
:         А
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

DstT0*

SrcT0
*(
_output_shapes
:         А
h
dnn/zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Н
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
а
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
_output_shapes
: 
л
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"А   @   *
_output_shapes
:
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *є5╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *є5>*
_output_shapes
: 
в
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes
:	А@*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
▒
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
г
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
╦
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes
:	А@*
dtype0*
shape:	А@*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
Ш
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А@
▓
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
▓
1dnn/hiddenlayer_1/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB@*    *
_output_shapes
:@
┐
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:@
к
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:@
v
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes
:	А@
╗
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         @
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:@
б
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*'
_output_shapes
:         @*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         @
]
dnn/zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ф
dnn/zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:         @
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         @
j
dnn/zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
а
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"@       *
_output_shapes
:
╣
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *  А╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *  А>*
_output_shapes
: 
б
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:@ *
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
Ю
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
в
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
╔
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:@ *
dtype0*
shape
:@ *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:@ 
▒
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
▓
1dnn/hiddenlayer_2/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB *    *
_output_shapes
: 
┐
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
: 
к
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
: 
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:@ 
╗
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:          
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
: 
б
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*'
_output_shapes
:          *
T0*
data_formatNHWC
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:          
]
dnn/zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ф
dnn/zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:          
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:          
j
dnn/zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
а
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_2/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
╣
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"       *
_output_shapes
:
л
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *JQ┌╛*
_output_shapes
: 
л
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *JQ┌>*
_output_shapes
: 
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
╗
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
√
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

: 
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
д
*dnn/logits/biases/part_0/Initializer/zerosConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
▒
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
ъ
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/zeros*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

: 
н
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:         *
T0*
data_formatNHWC
]
dnn/zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
В
dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:         
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
j
dnn/zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
dtype0*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
_output_shapes
: 
Я
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst*
dtype0**
value!B Bdnn/dnn/logits/activation*
_output_shapes
: 
Б
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
t
/linear/linear_model/alpha/to_sparse_input/ShapeShapebatch*
out_type0*
T0*
_output_shapes
:
Ы
.linear/linear_model/alpha/to_sparse_input/CastCast/linear/linear_model/alpha/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
s
2linear/linear_model/alpha/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
л
2linear/linear_model/alpha/to_sparse_input/NotEqualNotEqualbatch2linear/linear_model/alpha/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
Х
/linear/linear_model/alpha/to_sparse_input/WhereWhere2linear/linear_model/alpha/to_sparse_input/NotEqual*'
_output_shapes
:         
К
7linear/linear_model/alpha/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╕
1linear/linear_model/alpha/to_sparse_input/ReshapeReshapebatch7linear/linear_model/alpha/to_sparse_input/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
О
=linear/linear_model/alpha/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╪
7linear/linear_model/alpha/to_sparse_input/strided_sliceStridedSlice/linear/linear_model/alpha/to_sparse_input/Where=linear/linear_model/alpha/to_sparse_input/strided_slice/stack?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_1?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Т
Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Т
Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ф
9linear/linear_model/alpha/to_sparse_input/strided_slice_1StridedSlice/linear/linear_model/alpha/to_sparse_input/Where?linear/linear_model/alpha/to_sparse_input/strided_slice_1/stackAlinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_1Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
е
1linear/linear_model/alpha/to_sparse_input/unstackUnpack.linear/linear_model/alpha/to_sparse_input/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
ж
/linear/linear_model/alpha/to_sparse_input/stackPack3linear/linear_model/alpha/to_sparse_input/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
╥
-linear/linear_model/alpha/to_sparse_input/MulMul9linear/linear_model/alpha/to_sparse_input/strided_slice_1/linear/linear_model/alpha/to_sparse_input/stack*
T0	*'
_output_shapes
:         
Й
?linear/linear_model/alpha/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
я
-linear/linear_model/alpha/to_sparse_input/SumSum-linear/linear_model/alpha/to_sparse_input/Mul?linear/linear_model/alpha/to_sparse_input/Sum/reduction_indices*#
_output_shapes
:         *
T0	*
	keep_dims( *

Tidx0
╩
-linear/linear_model/alpha/to_sparse_input/AddAdd7linear/linear_model/alpha/to_sparse_input/strided_slice-linear/linear_model/alpha/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
°
0linear/linear_model/alpha/to_sparse_input/GatherGather1linear/linear_model/alpha/to_sparse_input/Reshape-linear/linear_model/alpha/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         

,linear/linear_model/alpha/alpha_lookup/ConstConst*
dtype0*
valueBBax01Bax02*
_output_shapes
:
m
+linear/linear_model/alpha/alpha_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
t
2linear/linear_model/alpha/alpha_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
t
2linear/linear_model/alpha/alpha_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Є
,linear/linear_model/alpha/alpha_lookup/rangeRange2linear/linear_model/alpha/alpha_lookup/range/start+linear/linear_model/alpha/alpha_lookup/Size2linear/linear_model/alpha/alpha_lookup/range/delta*

Tidx0*
_output_shapes
:
Ш
.linear/linear_model/alpha/alpha_lookup/ToInt64Cast,linear/linear_model/alpha/alpha_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
╜
1linear/linear_model/alpha/alpha_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
В
7linear/linear_model/alpha/alpha_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
·
<linear/linear_model/alpha/alpha_lookup/hash_table/table_initInitializeTableV21linear/linear_model/alpha/alpha_lookup/hash_table,linear/linear_model/alpha/alpha_lookup/Const.linear/linear_model/alpha/alpha_lookup/ToInt64*

Tkey0*

Tval0	
Ъ
+linear/linear_model/alpha/hash_table_LookupLookupTableFindV21linear/linear_model/alpha/alpha_lookup/hash_table0linear/linear_model/alpha/to_sparse_input/Gather7linear/linear_model/alpha/alpha_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
Р
$linear/linear_model/alpha/Shape/CastCast.linear/linear_model/alpha/to_sparse_input/Cast*

DstT0*

SrcT0	*
_output_shapes
:
w
-linear/linear_model/alpha/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
y
/linear/linear_model/alpha/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
y
/linear/linear_model/alpha/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
'linear/linear_model/alpha/strided_sliceStridedSlice$linear/linear_model/alpha/Shape/Cast-linear/linear_model/alpha/strided_slice/stack/linear/linear_model/alpha/strided_slice/stack_1/linear/linear_model/alpha/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
m
"linear/linear_model/alpha/Cast/x/1Const*
dtype0*
valueB :
         *
_output_shapes
: 
п
 linear/linear_model/alpha/Cast/xPack'linear/linear_model/alpha/strided_slice"linear/linear_model/alpha/Cast/x/1*
N*
T0*
_output_shapes
:*

axis 
|
linear/linear_model/alpha/CastCast linear/linear_model/alpha/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
ш
'linear/linear_model/alpha/SparseReshapeSparseReshape/linear/linear_model/alpha/to_sparse_input/Where.linear/linear_model/alpha/to_sparse_input/Castlinear/linear_model/alpha/Cast*-
_output_shapes
:         :
Ч
0linear/linear_model/alpha/SparseReshape/IdentityIdentity+linear/linear_model/alpha/hash_table_Lookup*
T0	*#
_output_shapes
:         
╠
:linear/linear_model/alpha/weights/part_0/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
valueB*    *
_output_shapes

:
┘
(linear/linear_model/alpha/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
shared_name 
о
/linear/linear_model/alpha/weights/part_0/AssignAssign(linear/linear_model/alpha/weights/part_0:linear/linear_model/alpha/weights/part_0/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
use_locking(*
T0*
_output_shapes

:
╔
-linear/linear_model/alpha/weights/part_0/readIdentity(linear/linear_model/alpha/weights/part_0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
T0*
_output_shapes

:
|
2linear/linear_model/alpha/weighted_sum/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
{
1linear/linear_model/alpha/weighted_sum/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
∙
,linear/linear_model/alpha/weighted_sum/SliceSlice)linear/linear_model/alpha/SparseReshape:12linear/linear_model/alpha/weighted_sum/Slice/begin1linear/linear_model/alpha/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
v
,linear/linear_model/alpha/weighted_sum/ConstConst*
dtype0*
valueB: *
_output_shapes
:
═
+linear/linear_model/alpha/weighted_sum/ProdProd,linear/linear_model/alpha/weighted_sum/Slice,linear/linear_model/alpha/weighted_sum/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
w
5linear/linear_model/alpha/weighted_sum/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
ш
-linear/linear_model/alpha/weighted_sum/GatherGather)linear/linear_model/alpha/SparseReshape:15linear/linear_model/alpha/weighted_sum/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
╦
-linear/linear_model/alpha/weighted_sum/Cast/xPack+linear/linear_model/alpha/weighted_sum/Prod-linear/linear_model/alpha/weighted_sum/Gather*
N*
T0	*
_output_shapes
:*

axis 
ў
4linear/linear_model/alpha/weighted_sum/SparseReshapeSparseReshape'linear/linear_model/alpha/SparseReshape)linear/linear_model/alpha/SparseReshape:1-linear/linear_model/alpha/weighted_sum/Cast/x*-
_output_shapes
:         :
й
=linear/linear_model/alpha/weighted_sum/SparseReshape/IdentityIdentity0linear/linear_model/alpha/SparseReshape/Identity*
T0	*#
_output_shapes
:         
w
5linear/linear_model/alpha/weighted_sum/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ч
3linear/linear_model/alpha/weighted_sum/GreaterEqualGreaterEqual=linear/linear_model/alpha/weighted_sum/SparseReshape/Identity5linear/linear_model/alpha/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:         
У
,linear/linear_model/alpha/weighted_sum/WhereWhere3linear/linear_model/alpha/weighted_sum/GreaterEqual*'
_output_shapes
:         
З
4linear/linear_model/alpha/weighted_sum/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
┘
.linear/linear_model/alpha/weighted_sum/ReshapeReshape,linear/linear_model/alpha/weighted_sum/Where4linear/linear_model/alpha/weighted_sum/Reshape/shape*#
_output_shapes
:         *
T0	*
Tshape0
 
/linear/linear_model/alpha/weighted_sum/Gather_1Gather4linear/linear_model/alpha/weighted_sum/SparseReshape.linear/linear_model/alpha/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Д
/linear/linear_model/alpha/weighted_sum/Gather_2Gather=linear/linear_model/alpha/weighted_sum/SparseReshape/Identity.linear/linear_model/alpha/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
Ш
/linear/linear_model/alpha/weighted_sum/IdentityIdentity6linear/linear_model/alpha/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
В
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ш
Nlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
П
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_sliceStridedSlice/linear/linear_model/alpha/weighted_sum/IdentityNlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stackPlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
┴
?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/CastCastHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╦
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/rangeRangeFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/start?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/CastFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╚
Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Cast_1Cast@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
б
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
д
Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1StridedSlice/linear/linear_model/alpha/weighted_sum/Gather_1Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackRlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
к
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffListDiffAlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Cast_1Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ь
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ь
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ч
Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2StridedSlice/linear/linear_model/alpha/weighted_sum/IdentityPlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackRlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ф
Ilinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Ы
Elinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims
ExpandDimsJlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2Ilinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Ш
Vlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ш
Vlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ы
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDenseSparseToDenseClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffElinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDimsVlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesVlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
Щ
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ь
Blinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ReshapeReshapeClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:         *
T0	*
Tshape0
╚
Elinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/zeros_like	ZerosLikeBlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
ч
Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concatConcatV2Blinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ReshapeElinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/zeros_likeFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
├
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ShapeShapeClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
∙
?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/FillFill@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Shape@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
К
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╘
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1ConcatV2/linear/linear_model/alpha/weighted_sum/Gather_1Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concatHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
К
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╬
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2ConcatV2/linear/linear_model/alpha/weighted_sum/Gather_2?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/FillHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:         *
T0	
╒
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorderSparseReorderClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/linear/linear_model/alpha/weighted_sum/Identity*
T0	*6
_output_shapes$
":         :         
е
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/IdentityIdentity/linear/linear_model/alpha/weighted_sum/Identity*
T0	*
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
е
Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
е
Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
┼
Llinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorderRlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stackTlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
╓
Clinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/CastCastLlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
ч
Elinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/UniqueUniqueJlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:         :         
ь
Olinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/embedding_lookupGather-linear/linear_model/alpha/weights/part_0/readElinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Unique*
validate_indices(*
Tparams0*
Tindices0	*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*'
_output_shapes
:         
я
>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparseSparseSegmentSumOlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/embedding_lookupGlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Unique:1Clinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:         *
T0*

Tidx0
З
6linear/linear_model/alpha/weighted_sum/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
¤
0linear/linear_model/alpha/weighted_sum/Reshape_1ReshapeHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense6linear/linear_model/alpha/weighted_sum/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
к
,linear/linear_model/alpha/weighted_sum/ShapeShape>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Д
:linear/linear_model/alpha/weighted_sum/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Ж
<linear/linear_model/alpha/weighted_sum/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ж
<linear/linear_model/alpha/weighted_sum/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╝
4linear/linear_model/alpha/weighted_sum/strided_sliceStridedSlice,linear/linear_model/alpha/weighted_sum/Shape:linear/linear_model/alpha/weighted_sum/strided_slice/stack<linear/linear_model/alpha/weighted_sum/strided_slice/stack_1<linear/linear_model/alpha/weighted_sum/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
p
.linear/linear_model/alpha/weighted_sum/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╘
,linear/linear_model/alpha/weighted_sum/stackPack.linear/linear_model/alpha/weighted_sum/stack/04linear/linear_model/alpha/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:*

axis 
р
+linear/linear_model/alpha/weighted_sum/TileTile0linear/linear_model/alpha/weighted_sum/Reshape_1,linear/linear_model/alpha/weighted_sum/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
░
1linear/linear_model/alpha/weighted_sum/zeros_like	ZerosLike>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
В
&linear/linear_model/alpha/weighted_sumSelect+linear/linear_model/alpha/weighted_sum/Tile1linear/linear_model/alpha/weighted_sum/zeros_like>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Ф
-linear/linear_model/alpha/weighted_sum/Cast_1Cast)linear/linear_model/alpha/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
~
4linear/linear_model/alpha/weighted_sum/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
}
3linear/linear_model/alpha/weighted_sum/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Г
.linear/linear_model/alpha/weighted_sum/Slice_1Slice-linear/linear_model/alpha/weighted_sum/Cast_14linear/linear_model/alpha/weighted_sum/Slice_1/begin3linear/linear_model/alpha/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ф
.linear/linear_model/alpha/weighted_sum/Shape_1Shape&linear/linear_model/alpha/weighted_sum*
out_type0*
T0*
_output_shapes
:
~
4linear/linear_model/alpha/weighted_sum/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Ж
3linear/linear_model/alpha/weighted_sum/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
Д
.linear/linear_model/alpha/weighted_sum/Slice_2Slice.linear/linear_model/alpha/weighted_sum/Shape_14linear/linear_model/alpha/weighted_sum/Slice_2/begin3linear/linear_model/alpha/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
t
2linear/linear_model/alpha/weighted_sum/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
З
-linear/linear_model/alpha/weighted_sum/concatConcatV2.linear/linear_model/alpha/weighted_sum/Slice_1.linear/linear_model/alpha/weighted_sum/Slice_22linear/linear_model/alpha/weighted_sum/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╥
0linear/linear_model/alpha/weighted_sum/Reshape_2Reshape&linear/linear_model/alpha/weighted_sum-linear/linear_model/alpha/weighted_sum/concat*'
_output_shapes
:         *
T0*
Tshape0
u
.linear/linear_model/beta/to_sparse_input/ShapeShapebatch:1*
out_type0*
T0*
_output_shapes
:
Щ
-linear/linear_model/beta/to_sparse_input/CastCast.linear/linear_model/beta/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
r
1linear/linear_model/beta/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
л
1linear/linear_model/beta/to_sparse_input/NotEqualNotEqualbatch:11linear/linear_model/beta/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
У
.linear/linear_model/beta/to_sparse_input/WhereWhere1linear/linear_model/beta/to_sparse_input/NotEqual*'
_output_shapes
:         
Й
6linear/linear_model/beta/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╕
0linear/linear_model/beta/to_sparse_input/ReshapeReshapebatch:16linear/linear_model/beta/to_sparse_input/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
Н
<linear/linear_model/beta/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
П
>linear/linear_model/beta/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
П
>linear/linear_model/beta/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╙
6linear/linear_model/beta/to_sparse_input/strided_sliceStridedSlice.linear/linear_model/beta/to_sparse_input/Where<linear/linear_model/beta/to_sparse_input/strided_slice/stack>linear/linear_model/beta/to_sparse_input/strided_slice/stack_1>linear/linear_model/beta/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
П
>linear/linear_model/beta/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
С
@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
С
@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
▀
8linear/linear_model/beta/to_sparse_input/strided_slice_1StridedSlice.linear/linear_model/beta/to_sparse_input/Where>linear/linear_model/beta/to_sparse_input/strided_slice_1/stack@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_1@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
г
0linear/linear_model/beta/to_sparse_input/unstackUnpack-linear/linear_model/beta/to_sparse_input/Cast*	
num*
T0	*
_output_shapes
: : *

axis 
д
.linear/linear_model/beta/to_sparse_input/stackPack2linear/linear_model/beta/to_sparse_input/unstack:1*
N*
T0	*
_output_shapes
:*

axis 
╧
,linear/linear_model/beta/to_sparse_input/MulMul8linear/linear_model/beta/to_sparse_input/strided_slice_1.linear/linear_model/beta/to_sparse_input/stack*
T0	*'
_output_shapes
:         
И
>linear/linear_model/beta/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
ь
,linear/linear_model/beta/to_sparse_input/SumSum,linear/linear_model/beta/to_sparse_input/Mul>linear/linear_model/beta/to_sparse_input/Sum/reduction_indices*#
_output_shapes
:         *
T0	*
	keep_dims( *

Tidx0
╟
,linear/linear_model/beta/to_sparse_input/AddAdd6linear/linear_model/beta/to_sparse_input/strided_slice,linear/linear_model/beta/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
ї
/linear/linear_model/beta/to_sparse_input/GatherGather0linear/linear_model/beta/to_sparse_input/Reshape,linear/linear_model/beta/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
}
*linear/linear_model/beta/beta_lookup/ConstConst*
dtype0*
valueBBbx01Bbx02*
_output_shapes
:
k
)linear/linear_model/beta/beta_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
r
0linear/linear_model/beta/beta_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
r
0linear/linear_model/beta/beta_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ъ
*linear/linear_model/beta/beta_lookup/rangeRange0linear/linear_model/beta/beta_lookup/range/start)linear/linear_model/beta/beta_lookup/Size0linear/linear_model/beta/beta_lookup/range/delta*

Tidx0*
_output_shapes
:
Ф
,linear/linear_model/beta/beta_lookup/ToInt64Cast*linear/linear_model/beta/beta_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
╗
/linear/linear_model/beta/beta_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
А
5linear/linear_model/beta/beta_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Є
:linear/linear_model/beta/beta_lookup/hash_table/table_initInitializeTableV2/linear/linear_model/beta/beta_lookup/hash_table*linear/linear_model/beta/beta_lookup/Const,linear/linear_model/beta/beta_lookup/ToInt64*

Tkey0*

Tval0	
Ф
*linear/linear_model/beta/hash_table_LookupLookupTableFindV2/linear/linear_model/beta/beta_lookup/hash_table/linear/linear_model/beta/to_sparse_input/Gather5linear/linear_model/beta/beta_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
О
#linear/linear_model/beta/Shape/CastCast-linear/linear_model/beta/to_sparse_input/Cast*

DstT0*

SrcT0	*
_output_shapes
:
v
,linear/linear_model/beta/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
x
.linear/linear_model/beta/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
x
.linear/linear_model/beta/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
√
&linear/linear_model/beta/strided_sliceStridedSlice#linear/linear_model/beta/Shape/Cast,linear/linear_model/beta/strided_slice/stack.linear/linear_model/beta/strided_slice/stack_1.linear/linear_model/beta/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
l
!linear/linear_model/beta/Cast/x/1Const*
dtype0*
valueB :
         *
_output_shapes
: 
м
linear/linear_model/beta/Cast/xPack&linear/linear_model/beta/strided_slice!linear/linear_model/beta/Cast/x/1*
N*
T0*
_output_shapes
:*

axis 
z
linear/linear_model/beta/CastCastlinear/linear_model/beta/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
ф
&linear/linear_model/beta/SparseReshapeSparseReshape.linear/linear_model/beta/to_sparse_input/Where-linear/linear_model/beta/to_sparse_input/Castlinear/linear_model/beta/Cast*-
_output_shapes
:         :
Х
/linear/linear_model/beta/SparseReshape/IdentityIdentity*linear/linear_model/beta/hash_table_Lookup*
T0	*#
_output_shapes
:         
╩
9linear/linear_model/beta/weights/part_0/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
valueB*    *
_output_shapes

:
╫
'linear/linear_model/beta/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
shared_name 
к
.linear/linear_model/beta/weights/part_0/AssignAssign'linear/linear_model/beta/weights/part_09linear/linear_model/beta/weights/part_0/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
use_locking(*
T0*
_output_shapes

:
╞
,linear/linear_model/beta/weights/part_0/readIdentity'linear/linear_model/beta/weights/part_0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
T0*
_output_shapes

:
{
1linear/linear_model/beta/weighted_sum/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
z
0linear/linear_model/beta/weighted_sum/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ї
+linear/linear_model/beta/weighted_sum/SliceSlice(linear/linear_model/beta/SparseReshape:11linear/linear_model/beta/weighted_sum/Slice/begin0linear/linear_model/beta/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
u
+linear/linear_model/beta/weighted_sum/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╩
*linear/linear_model/beta/weighted_sum/ProdProd+linear/linear_model/beta/weighted_sum/Slice+linear/linear_model/beta/weighted_sum/Const*
_output_shapes
: *
T0	*
	keep_dims( *

Tidx0
v
4linear/linear_model/beta/weighted_sum/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
х
,linear/linear_model/beta/weighted_sum/GatherGather(linear/linear_model/beta/SparseReshape:14linear/linear_model/beta/weighted_sum/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
╚
,linear/linear_model/beta/weighted_sum/Cast/xPack*linear/linear_model/beta/weighted_sum/Prod,linear/linear_model/beta/weighted_sum/Gather*
N*
T0	*
_output_shapes
:*

axis 
є
3linear/linear_model/beta/weighted_sum/SparseReshapeSparseReshape&linear/linear_model/beta/SparseReshape(linear/linear_model/beta/SparseReshape:1,linear/linear_model/beta/weighted_sum/Cast/x*-
_output_shapes
:         :
з
<linear/linear_model/beta/weighted_sum/SparseReshape/IdentityIdentity/linear/linear_model/beta/SparseReshape/Identity*
T0	*#
_output_shapes
:         
v
4linear/linear_model/beta/weighted_sum/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ф
2linear/linear_model/beta/weighted_sum/GreaterEqualGreaterEqual<linear/linear_model/beta/weighted_sum/SparseReshape/Identity4linear/linear_model/beta/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:         
С
+linear/linear_model/beta/weighted_sum/WhereWhere2linear/linear_model/beta/weighted_sum/GreaterEqual*'
_output_shapes
:         
Ж
3linear/linear_model/beta/weighted_sum/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╓
-linear/linear_model/beta/weighted_sum/ReshapeReshape+linear/linear_model/beta/weighted_sum/Where3linear/linear_model/beta/weighted_sum/Reshape/shape*#
_output_shapes
:         *
T0	*
Tshape0
№
.linear/linear_model/beta/weighted_sum/Gather_1Gather3linear/linear_model/beta/weighted_sum/SparseReshape-linear/linear_model/beta/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Б
.linear/linear_model/beta/weighted_sum/Gather_2Gather<linear/linear_model/beta/weighted_sum/SparseReshape/Identity-linear/linear_model/beta/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
Ц
.linear/linear_model/beta/weighted_sum/IdentityIdentity5linear/linear_model/beta/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
Б
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ч
Mlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
К
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_sliceStridedSlice.linear/linear_model/beta/weighted_sum/IdentityMlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stackOlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
┐
>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/CastCastGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╟
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/rangeRangeElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/start>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/CastElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╞
@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Cast_1Cast?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
а
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Я
Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1StridedSlice.linear/linear_model/beta/weighted_sum/Gather_1Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackQlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
з
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffListDiff@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Cast_1Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ы
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ы
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Т
Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2StridedSlice.linear/linear_model/beta/weighted_sum/IdentityOlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackQlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
У
Hlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Ш
Dlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims
ExpandDimsIlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2Hlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Ч
Ulinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ч
Ulinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ц
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDenseSparseToDenseBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffDlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDimsUlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesUlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
Ш
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Щ
Alinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ReshapeReshapeBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape/shape*'
_output_shapes
:         *
T0	*
Tshape0
╞
Dlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/zeros_like	ZerosLikeAlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
у
@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concatConcatV2Alinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ReshapeDlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/zeros_likeElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
┴
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ShapeShapeBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ў
>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/FillFill?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Shape?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
Й
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╨
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1ConcatV2.linear/linear_model/beta/weighted_sum/Gather_1@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concatGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1/axis*
N*

Tidx0*'
_output_shapes
:         *
T0	
Й
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╩
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2ConcatV2.linear/linear_model/beta/weighted_sum/Gather_2>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/FillGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2/axis*
N*

Tidx0*#
_output_shapes
:         *
T0	
╤
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorderSparseReorderBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2.linear/linear_model/beta/weighted_sum/Identity*
T0	*6
_output_shapes$
":         :         
г
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/IdentityIdentity.linear/linear_model/beta/weighted_sum/Identity*
T0	*
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
д
Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
д
Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
└
Klinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorderQlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stackSlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
╘
Blinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/CastCastKlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
х
Dlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/UniqueUniqueIlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:         :         
ш
Nlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/embedding_lookupGather,linear/linear_model/beta/weights/part_0/readDlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Unique*
validate_indices(*
Tparams0*
Tindices0	*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*'
_output_shapes
:         
ы
=linear/linear_model/beta/weighted_sum/embedding_lookup_sparseSparseSegmentSumNlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/embedding_lookupFlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Unique:1Blinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:         *
T0*

Tidx0
Ж
5linear/linear_model/beta/weighted_sum/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
·
/linear/linear_model/beta/weighted_sum/Reshape_1ReshapeGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense5linear/linear_model/beta/weighted_sum/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
и
+linear/linear_model/beta/weighted_sum/ShapeShape=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Г
9linear/linear_model/beta/weighted_sum/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Е
;linear/linear_model/beta/weighted_sum/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Е
;linear/linear_model/beta/weighted_sum/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╖
3linear/linear_model/beta/weighted_sum/strided_sliceStridedSlice+linear/linear_model/beta/weighted_sum/Shape9linear/linear_model/beta/weighted_sum/strided_slice/stack;linear/linear_model/beta/weighted_sum/strided_slice/stack_1;linear/linear_model/beta/weighted_sum/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
o
-linear/linear_model/beta/weighted_sum/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╤
+linear/linear_model/beta/weighted_sum/stackPack-linear/linear_model/beta/weighted_sum/stack/03linear/linear_model/beta/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:*

axis 
▌
*linear/linear_model/beta/weighted_sum/TileTile/linear/linear_model/beta/weighted_sum/Reshape_1+linear/linear_model/beta/weighted_sum/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
о
0linear/linear_model/beta/weighted_sum/zeros_like	ZerosLike=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
■
%linear/linear_model/beta/weighted_sumSelect*linear/linear_model/beta/weighted_sum/Tile0linear/linear_model/beta/weighted_sum/zeros_like=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Т
,linear/linear_model/beta/weighted_sum/Cast_1Cast(linear/linear_model/beta/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
}
3linear/linear_model/beta/weighted_sum/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
|
2linear/linear_model/beta/weighted_sum/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
 
-linear/linear_model/beta/weighted_sum/Slice_1Slice,linear/linear_model/beta/weighted_sum/Cast_13linear/linear_model/beta/weighted_sum/Slice_1/begin2linear/linear_model/beta/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
Т
-linear/linear_model/beta/weighted_sum/Shape_1Shape%linear/linear_model/beta/weighted_sum*
out_type0*
T0*
_output_shapes
:
}
3linear/linear_model/beta/weighted_sum/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Е
2linear/linear_model/beta/weighted_sum/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
А
-linear/linear_model/beta/weighted_sum/Slice_2Slice-linear/linear_model/beta/weighted_sum/Shape_13linear/linear_model/beta/weighted_sum/Slice_2/begin2linear/linear_model/beta/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
s
1linear/linear_model/beta/weighted_sum/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Г
,linear/linear_model/beta/weighted_sum/concatConcatV2-linear/linear_model/beta/weighted_sum/Slice_1-linear/linear_model/beta/weighted_sum/Slice_21linear/linear_model/beta/weighted_sum/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╧
/linear/linear_model/beta/weighted_sum/Reshape_2Reshape%linear/linear_model/beta/weighted_sum,linear/linear_model/beta/weighted_sum/concat*'
_output_shapes
:         *
T0*
Tshape0
╬
(linear/linear_model/weighted_sum_no_biasAddN0linear/linear_model/alpha/weighted_sum/Reshape_2/linear/linear_model/beta/weighted_sum/Reshape_2*'
_output_shapes
:         *
T0*
N
┬
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
valueB*    *
_output_shapes
:
╧
'linear/linear_model/bias_weights/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
shared_name 
ж
.linear/linear_model/bias_weights/part_0/AssignAssign'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
use_locking(*
T0*
_output_shapes
:
┬
,linear/linear_model/bias_weights/part_0/readIdentity'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
T0*
_output_shapes
:

 linear/linear_model/bias_weightsIdentity,linear/linear_model/bias_weights/part_0/read*
T0*
_output_shapes
:
└
 linear/linear_model/weighted_sumBiasAdd(linear/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*'
_output_shapes
:         *
T0*
data_formatNHWC
^
linear/zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Т
linear/zero_fraction/EqualEqual linear/linear_model/weighted_sumlinear/zero_fraction/zero*
T0*'
_output_shapes
:         
~
linear/zero_fraction/CastCastlinear/zero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
k
linear/zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Ц
linear/zero_fraction/MeanMeanlinear/zero_fraction/Castlinear/zero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Р
*linear/linear/fraction_of_zero_values/tagsConst*
dtype0*6
value-B+ B%linear/linear/fraction_of_zero_values*
_output_shapes
: 
Ю
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction/Mean*
T0*
_output_shapes
: 
u
linear/linear/activation/tagConst*
dtype0*)
value B Blinear/linear/activation*
_output_shapes
: 
Н
linear/linear/activationHistogramSummarylinear/linear/activation/tag linear/linear_model/weighted_sum*
T0*
_output_shapes
: 
r
addAdddnn/logits/BiasAdd linear/linear_model/weighted_sum*
T0*'
_output_shapes
:         
o
+binary_logistic_head/predictions/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:         
n
,binary_logistic_head/predictions/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
┌
'binary_logistic_head/predictions/concatConcatV2+binary_logistic_head/predictions/zeros_likeadd,binary_logistic_head/predictions/concat/axis*
N*

Tidx0*'
_output_shapes
:         *
T0
k
)binary_logistic_head/predictions/logisticSigmoidadd*
T0*'
_output_shapes
:         
Ф
.binary_logistic_head/predictions/probabilitiesSoftmax'binary_logistic_head/predictions/concat*
T0*'
_output_shapes
:         
t
2binary_logistic_head/predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
╔
(binary_logistic_head/predictions/classesArgMax'binary_logistic_head/predictions/concat2binary_logistic_head/predictions/classes/dimension*#
_output_shapes
:         *
T0*

Tidx0
Т
6binary_logistic_head/log_loss_with_two_classes/ToFloatCasthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
}
9binary_logistic_head/log_loss_with_two_classes/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:         
╜
;binary_logistic_head/log_loss_with_two_classes/GreaterEqualGreaterEqualadd9binary_logistic_head/log_loss_with_two_classes/zeros_like*
T0*'
_output_shapes
:         
ю
5binary_logistic_head/log_loss_with_two_classes/SelectSelect;binary_logistic_head/log_loss_with_two_classes/GreaterEqualadd9binary_logistic_head/log_loss_with_two_classes/zeros_like*
T0*'
_output_shapes
:         
p
2binary_logistic_head/log_loss_with_two_classes/NegNegadd*
T0*'
_output_shapes
:         
щ
7binary_logistic_head/log_loss_with_two_classes/Select_1Select;binary_logistic_head/log_loss_with_two_classes/GreaterEqual2binary_logistic_head/log_loss_with_two_classes/Negadd*
T0*'
_output_shapes
:         
и
2binary_logistic_head/log_loss_with_two_classes/mulMuladd6binary_logistic_head/log_loss_with_two_classes/ToFloat*
T0*'
_output_shapes
:         
╓
2binary_logistic_head/log_loss_with_two_classes/subSub5binary_logistic_head/log_loss_with_two_classes/Select2binary_logistic_head/log_loss_with_two_classes/mul*
T0*'
_output_shapes
:         
д
2binary_logistic_head/log_loss_with_two_classes/ExpExp7binary_logistic_head/log_loss_with_two_classes/Select_1*
T0*'
_output_shapes
:         
г
4binary_logistic_head/log_loss_with_two_classes/Log1pLog1p2binary_logistic_head/log_loss_with_two_classes/Exp*
T0*'
_output_shapes
:         
╤
.binary_logistic_head/log_loss_with_two_classesAdd2binary_logistic_head/log_loss_with_two_classes/sub4binary_logistic_head/log_loss_with_two_classes/Log1p*
T0*'
_output_shapes
:         
К
9binary_logistic_head/log_loss_with_two_classes/loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
ф
3binary_logistic_head/log_loss_with_two_classes/lossMean.binary_logistic_head/log_loss_with_two_classes9binary_logistic_head/log_loss_with_two_classes/loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
l
'binary_logistic_head/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
▓
"binary_logistic_head/ScalarSummaryScalarSummary'binary_logistic_head/ScalarSummary/tags3binary_logistic_head/log_loss_with_two_classes/loss*
T0*
_output_shapes
: 
l
'binary_logistic_head/metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
'binary_logistic_head/metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Р
.binary_logistic_head/metrics/mean/total/AssignAssign'binary_logistic_head/metrics/mean/total'binary_logistic_head/metrics/mean/zeros*
validate_shape(*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 
╛
,binary_logistic_head/metrics/mean/total/readIdentity'binary_logistic_head/metrics/mean/total*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
'binary_logistic_head/metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Т
.binary_logistic_head/metrics/mean/count/AssignAssign'binary_logistic_head/metrics/mean/count)binary_logistic_head/metrics/mean/zeros_1*
validate_shape(*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 
╛
,binary_logistic_head/metrics/mean/count/readIdentity'binary_logistic_head/metrics/mean/count*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
T0*
_output_shapes
: 
h
&binary_logistic_head/metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Л
+binary_logistic_head/metrics/mean/ToFloat_1Cast&binary_logistic_head/metrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
j
'binary_logistic_head/metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
╚
%binary_logistic_head/metrics/mean/SumSum3binary_logistic_head/log_loss_with_two_classes/loss'binary_logistic_head/metrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
°
+binary_logistic_head/metrics/mean/AssignAdd	AssignAdd'binary_logistic_head/metrics/mean/total%binary_logistic_head/metrics/mean/Sum*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
╢
-binary_logistic_head/metrics/mean/AssignAdd_1	AssignAdd'binary_logistic_head/metrics/mean/count+binary_logistic_head/metrics/mean/ToFloat_14^binary_logistic_head/log_loss_with_two_classes/loss*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
░
)binary_logistic_head/metrics/mean/GreaterGreater,binary_logistic_head/metrics/mean/count/read+binary_logistic_head/metrics/mean/Greater/y*
T0*
_output_shapes
: 
▒
)binary_logistic_head/metrics/mean/truedivRealDiv,binary_logistic_head/metrics/mean/total/read,binary_logistic_head/metrics/mean/count/read*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╙
'binary_logistic_head/metrics/mean/valueSelect)binary_logistic_head/metrics/mean/Greater)binary_logistic_head/metrics/mean/truediv)binary_logistic_head/metrics/mean/value/e*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╡
+binary_logistic_head/metrics/mean/Greater_1Greater-binary_logistic_head/metrics/mean/AssignAdd_1-binary_logistic_head/metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
│
+binary_logistic_head/metrics/mean/truediv_1RealDiv+binary_logistic_head/metrics/mean/AssignAdd-binary_logistic_head/metrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
▀
+binary_logistic_head/metrics/mean/update_opSelect+binary_logistic_head/metrics/mean/Greater_1+binary_logistic_head/metrics/mean/truediv_1-binary_logistic_head/metrics/mean/update_op/e*
T0*
_output_shapes
: 
н
Abinary_logistic_head/metrics/remove_squeezable_dimensions/SqueezeSqueezehash_table_Lookup*
squeeze_dims

         *
T0	*#
_output_shapes
:         
╞
"binary_logistic_head/metrics/EqualEqual(binary_logistic_head/predictions/classesAbinary_logistic_head/metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
Н
$binary_logistic_head/metrics/ToFloatCast"binary_logistic_head/metrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:         
p
+binary_logistic_head/metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
П
+binary_logistic_head/metrics/accuracy/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
а
2binary_logistic_head/metrics/accuracy/total/AssignAssign+binary_logistic_head/metrics/accuracy/total+binary_logistic_head/metrics/accuracy/zeros*
validate_shape(*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
╩
0binary_logistic_head/metrics/accuracy/total/readIdentity+binary_logistic_head/metrics/accuracy/total*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
П
+binary_logistic_head/metrics/accuracy/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
в
2binary_logistic_head/metrics/accuracy/count/AssignAssign+binary_logistic_head/metrics/accuracy/count-binary_logistic_head/metrics/accuracy/zeros_1*
validate_shape(*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
╩
0binary_logistic_head/metrics/accuracy/count/readIdentity+binary_logistic_head/metrics/accuracy/count*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
T0*
_output_shapes
: 
Й
*binary_logistic_head/metrics/accuracy/SizeSize$binary_logistic_head/metrics/ToFloat*
out_type0*
T0*
_output_shapes
: 
У
/binary_logistic_head/metrics/accuracy/ToFloat_1Cast*binary_logistic_head/metrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
u
+binary_logistic_head/metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
┴
)binary_logistic_head/metrics/accuracy/SumSum$binary_logistic_head/metrics/ToFloat+binary_logistic_head/metrics/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
И
/binary_logistic_head/metrics/accuracy/AssignAdd	AssignAdd+binary_logistic_head/metrics/accuracy/total)binary_logistic_head/metrics/accuracy/Sum*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
╖
1binary_logistic_head/metrics/accuracy/AssignAdd_1	AssignAdd+binary_logistic_head/metrics/accuracy/count/binary_logistic_head/metrics/accuracy/ToFloat_1%^binary_logistic_head/metrics/ToFloat*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╝
-binary_logistic_head/metrics/accuracy/GreaterGreater0binary_logistic_head/metrics/accuracy/count/read/binary_logistic_head/metrics/accuracy/Greater/y*
T0*
_output_shapes
: 
╜
-binary_logistic_head/metrics/accuracy/truedivRealDiv0binary_logistic_head/metrics/accuracy/total/read0binary_logistic_head/metrics/accuracy/count/read*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
у
+binary_logistic_head/metrics/accuracy/valueSelect-binary_logistic_head/metrics/accuracy/Greater-binary_logistic_head/metrics/accuracy/truediv-binary_logistic_head/metrics/accuracy/value/e*
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┴
/binary_logistic_head/metrics/accuracy/Greater_1Greater1binary_logistic_head/metrics/accuracy/AssignAdd_11binary_logistic_head/metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
┐
/binary_logistic_head/metrics/accuracy/truediv_1RealDiv/binary_logistic_head/metrics/accuracy/AssignAdd1binary_logistic_head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
я
/binary_logistic_head/metrics/accuracy/update_opSelect/binary_logistic_head/metrics/accuracy/Greater_1/binary_logistic_head/metrics/accuracy/truediv_11binary_logistic_head/metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean_1/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_1/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_1/total/AssignAssign)binary_logistic_head/metrics/mean_1/total)binary_logistic_head/metrics/mean_1/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_1/total/readIdentity)binary_logistic_head/metrics/mean_1/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_1/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_1/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_1/count/AssignAssign)binary_logistic_head/metrics/mean_1/count+binary_logistic_head/metrics/mean_1/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_1/count/readIdentity)binary_logistic_head/metrics/mean_1/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
T0*
_output_shapes
: 
М
(binary_logistic_head/metrics/mean_1/SizeSize)binary_logistic_head/predictions/logistic*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_1/ToFloat_1Cast(binary_logistic_head/metrics/mean_1/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┬
'binary_logistic_head/metrics/mean_1/SumSum)binary_logistic_head/predictions/logistic)binary_logistic_head/metrics/mean_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
А
-binary_logistic_head/metrics/mean_1/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_1/total'binary_logistic_head/metrics/mean_1/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
use_locking( *
T0*
_output_shapes
: 
┤
/binary_logistic_head/metrics/mean_1/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_1/count-binary_logistic_head/metrics/mean_1/ToFloat_1*^binary_logistic_head/predictions/logistic*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_1/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_1/GreaterGreater.binary_logistic_head/metrics/mean_1/count/read-binary_logistic_head/metrics/mean_1/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_1/truedivRealDiv.binary_logistic_head/metrics/mean_1/total/read.binary_logistic_head/metrics/mean_1/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_1/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_1/valueSelect+binary_logistic_head/metrics/mean_1/Greater+binary_logistic_head/metrics/mean_1/truediv+binary_logistic_head/metrics/mean_1/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_1/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_1/Greater_1Greater/binary_logistic_head/metrics/mean_1/AssignAdd_1/binary_logistic_head/metrics/mean_1/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_1/truediv_1RealDiv-binary_logistic_head/metrics/mean_1/AssignAdd/binary_logistic_head/metrics/mean_1/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_1/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_1/update_opSelect-binary_logistic_head/metrics/mean_1/Greater_1-binary_logistic_head/metrics/mean_1/truediv_1/binary_logistic_head/metrics/mean_1/update_op/e*
T0*
_output_shapes
: 
В
&binary_logistic_head/metrics/ToFloat_2Casthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
n
)binary_logistic_head/metrics/mean_2/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_2/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_2/total/AssignAssign)binary_logistic_head/metrics/mean_2/total)binary_logistic_head/metrics/mean_2/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_2/total/readIdentity)binary_logistic_head/metrics/mean_2/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_2/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_2/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_2/count/AssignAssign)binary_logistic_head/metrics/mean_2/count+binary_logistic_head/metrics/mean_2/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_2/count/readIdentity)binary_logistic_head/metrics/mean_2/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
T0*
_output_shapes
: 
Й
(binary_logistic_head/metrics/mean_2/SizeSize&binary_logistic_head/metrics/ToFloat_2*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_2/ToFloat_1Cast(binary_logistic_head/metrics/mean_2/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┐
'binary_logistic_head/metrics/mean_2/SumSum&binary_logistic_head/metrics/ToFloat_2)binary_logistic_head/metrics/mean_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
А
-binary_logistic_head/metrics/mean_2/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_2/total'binary_logistic_head/metrics/mean_2/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
use_locking( *
T0*
_output_shapes
: 
▒
/binary_logistic_head/metrics/mean_2/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_2/count-binary_logistic_head/metrics/mean_2/ToFloat_1'^binary_logistic_head/metrics/ToFloat_2*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_2/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_2/GreaterGreater.binary_logistic_head/metrics/mean_2/count/read-binary_logistic_head/metrics/mean_2/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_2/truedivRealDiv.binary_logistic_head/metrics/mean_2/total/read.binary_logistic_head/metrics/mean_2/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_2/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_2/valueSelect+binary_logistic_head/metrics/mean_2/Greater+binary_logistic_head/metrics/mean_2/truediv+binary_logistic_head/metrics/mean_2/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_2/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_2/Greater_1Greater/binary_logistic_head/metrics/mean_2/AssignAdd_1/binary_logistic_head/metrics/mean_2/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_2/truediv_1RealDiv-binary_logistic_head/metrics/mean_2/AssignAdd/binary_logistic_head/metrics/mean_2/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_2/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_2/update_opSelect-binary_logistic_head/metrics/mean_2/Greater_1-binary_logistic_head/metrics/mean_2/truediv_1/binary_logistic_head/metrics/mean_2/update_op/e*
T0*
_output_shapes
: 
В
&binary_logistic_head/metrics/ToFloat_3Casthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
n
)binary_logistic_head/metrics/mean_3/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_3/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_3/total/AssignAssign)binary_logistic_head/metrics/mean_3/total)binary_logistic_head/metrics/mean_3/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_3/total/readIdentity)binary_logistic_head/metrics/mean_3/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_3/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_3/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_3/count/AssignAssign)binary_logistic_head/metrics/mean_3/count+binary_logistic_head/metrics/mean_3/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_3/count/readIdentity)binary_logistic_head/metrics/mean_3/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
T0*
_output_shapes
: 
Й
(binary_logistic_head/metrics/mean_3/SizeSize&binary_logistic_head/metrics/ToFloat_3*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_3/ToFloat_1Cast(binary_logistic_head/metrics/mean_3/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┐
'binary_logistic_head/metrics/mean_3/SumSum&binary_logistic_head/metrics/ToFloat_3)binary_logistic_head/metrics/mean_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
А
-binary_logistic_head/metrics/mean_3/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_3/total'binary_logistic_head/metrics/mean_3/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
use_locking( *
T0*
_output_shapes
: 
▒
/binary_logistic_head/metrics/mean_3/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_3/count-binary_logistic_head/metrics/mean_3/ToFloat_1'^binary_logistic_head/metrics/ToFloat_3*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_3/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_3/GreaterGreater.binary_logistic_head/metrics/mean_3/count/read-binary_logistic_head/metrics/mean_3/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_3/truedivRealDiv.binary_logistic_head/metrics/mean_3/total/read.binary_logistic_head/metrics/mean_3/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_3/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_3/valueSelect+binary_logistic_head/metrics/mean_3/Greater+binary_logistic_head/metrics/mean_3/truediv+binary_logistic_head/metrics/mean_3/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_3/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_3/Greater_1Greater/binary_logistic_head/metrics/mean_3/AssignAdd_1/binary_logistic_head/metrics/mean_3/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_3/truediv_1RealDiv-binary_logistic_head/metrics/mean_3/AssignAdd/binary_logistic_head/metrics/mean_3/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_3/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_3/update_opSelect-binary_logistic_head/metrics/mean_3/Greater_1-binary_logistic_head/metrics/mean_3/truediv_1/binary_logistic_head/metrics/mean_3/update_op/e*
T0*
_output_shapes
: 
}
!binary_logistic_head/metrics/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         

.binary_logistic_head/metrics/auc/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╬
(binary_logistic_head/metrics/auc/ReshapeReshape)binary_logistic_head/predictions/logistic.binary_logistic_head/metrics/auc/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
Б
0binary_logistic_head/metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╩
*binary_logistic_head/metrics/auc/Reshape_1Reshape!binary_logistic_head/metrics/Cast0binary_logistic_head/metrics/auc/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
О
&binary_logistic_head/metrics/auc/ShapeShape(binary_logistic_head/metrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
~
4binary_logistic_head/metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
А
6binary_logistic_head/metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
А
6binary_logistic_head/metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ю
.binary_logistic_head/metrics/auc/strided_sliceStridedSlice&binary_logistic_head/metrics/auc/Shape4binary_logistic_head/metrics/auc/strided_slice/stack6binary_logistic_head/metrics/auc/strided_slice/stack_16binary_logistic_head/metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Х
&binary_logistic_head/metrics/auc/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
y
/binary_logistic_head/metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
╚
+binary_logistic_head/metrics/auc/ExpandDims
ExpandDims&binary_logistic_head/metrics/auc/Const/binary_logistic_head/metrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
j
(binary_logistic_head/metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
┬
&binary_logistic_head/metrics/auc/stackPack(binary_logistic_head/metrics/auc/stack/0.binary_logistic_head/metrics/auc/strided_slice*
N*
T0*
_output_shapes
:*

axis 
╟
%binary_logistic_head/metrics/auc/TileTile+binary_logistic_head/metrics/auc/ExpandDims&binary_logistic_head/metrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
В
/binary_logistic_head/metrics/auc/transpose/RankRank(binary_logistic_head/metrics/auc/Reshape*
T0*
_output_shapes
: 
r
0binary_logistic_head/metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
╣
.binary_logistic_head/metrics/auc/transpose/subSub/binary_logistic_head/metrics/auc/transpose/Rank0binary_logistic_head/metrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
x
6binary_logistic_head/metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
x
6binary_logistic_head/metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
В
0binary_logistic_head/metrics/auc/transpose/RangeRange6binary_logistic_head/metrics/auc/transpose/Range/start/binary_logistic_head/metrics/auc/transpose/Rank6binary_logistic_head/metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:
╛
0binary_logistic_head/metrics/auc/transpose/sub_1Sub.binary_logistic_head/metrics/auc/transpose/sub0binary_logistic_head/metrics/auc/transpose/Range*
T0*
_output_shapes
:
╥
*binary_logistic_head/metrics/auc/transpose	Transpose(binary_logistic_head/metrics/auc/Reshape0binary_logistic_head/metrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
В
1binary_logistic_head/metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
╙
'binary_logistic_head/metrics/auc/Tile_1Tile*binary_logistic_head/metrics/auc/transpose1binary_logistic_head/metrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
╢
(binary_logistic_head/metrics/auc/GreaterGreater'binary_logistic_head/metrics/auc/Tile_1%binary_logistic_head/metrics/auc/Tile*
T0*(
_output_shapes
:╚         
Н
+binary_logistic_head/metrics/auc/LogicalNot
LogicalNot(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
В
1binary_logistic_head/metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
╙
'binary_logistic_head/metrics/auc/Tile_2Tile*binary_logistic_head/metrics/auc/Reshape_11binary_logistic_head/metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
О
-binary_logistic_head/metrics/auc/LogicalNot_1
LogicalNot'binary_logistic_head/metrics/auc/Tile_2*(
_output_shapes
:╚         
u
&binary_logistic_head/metrics/auc/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Э
/binary_logistic_head/metrics/auc/true_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
м
6binary_logistic_head/metrics/auc/true_positives/AssignAssign/binary_logistic_head/metrics/auc/true_positives&binary_logistic_head/metrics/auc/zeros*
validate_shape(*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
█
4binary_logistic_head/metrics/auc/true_positives/readIdentity/binary_logistic_head/metrics/auc/true_positives*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
T0*
_output_shapes	
:╚
╢
+binary_logistic_head/metrics/auc/LogicalAnd
LogicalAnd'binary_logistic_head/metrics/auc/Tile_2(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
б
*binary_logistic_head/metrics/auc/ToFloat_1Cast+binary_logistic_head/metrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
x
6binary_logistic_head/metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╥
$binary_logistic_head/metrics/auc/SumSum*binary_logistic_head/metrics/auc/ToFloat_16binary_logistic_head/metrics/auc/Sum/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
Л
*binary_logistic_head/metrics/auc/AssignAdd	AssignAdd/binary_logistic_head/metrics/auc/true_positives$binary_logistic_head/metrics/auc/Sum*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Ю
0binary_logistic_head/metrics/auc/false_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
▒
7binary_logistic_head/metrics/auc/false_negatives/AssignAssign0binary_logistic_head/metrics/auc/false_negatives(binary_logistic_head/metrics/auc/zeros_1*
validate_shape(*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
▐
5binary_logistic_head/metrics/auc/false_negatives/readIdentity0binary_logistic_head/metrics/auc/false_negatives*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
T0*
_output_shapes	
:╚
╗
-binary_logistic_head/metrics/auc/LogicalAnd_1
LogicalAnd'binary_logistic_head/metrics/auc/Tile_2+binary_logistic_head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_2Cast-binary_logistic_head/metrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_1Sum*binary_logistic_head/metrics/auc/ToFloat_28binary_logistic_head/metrics/auc/Sum_1/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
С
,binary_logistic_head/metrics/auc/AssignAdd_1	AssignAdd0binary_logistic_head/metrics/auc/false_negatives&binary_logistic_head/metrics/auc/Sum_1*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Э
/binary_logistic_head/metrics/auc/true_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
о
6binary_logistic_head/metrics/auc/true_negatives/AssignAssign/binary_logistic_head/metrics/auc/true_negatives(binary_logistic_head/metrics/auc/zeros_2*
validate_shape(*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
█
4binary_logistic_head/metrics/auc/true_negatives/readIdentity/binary_logistic_head/metrics/auc/true_negatives*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
T0*
_output_shapes	
:╚
┴
-binary_logistic_head/metrics/auc/LogicalAnd_2
LogicalAnd-binary_logistic_head/metrics/auc/LogicalNot_1+binary_logistic_head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_3Cast-binary_logistic_head/metrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_2Sum*binary_logistic_head/metrics/auc/ToFloat_38binary_logistic_head/metrics/auc/Sum_2/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
П
,binary_logistic_head/metrics/auc/AssignAdd_2	AssignAdd/binary_logistic_head/metrics/auc/true_negatives&binary_logistic_head/metrics/auc/Sum_2*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Ю
0binary_logistic_head/metrics/auc/false_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
▒
7binary_logistic_head/metrics/auc/false_positives/AssignAssign0binary_logistic_head/metrics/auc/false_positives(binary_logistic_head/metrics/auc/zeros_3*
validate_shape(*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
▐
5binary_logistic_head/metrics/auc/false_positives/readIdentity0binary_logistic_head/metrics/auc/false_positives*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
T0*
_output_shapes	
:╚
╛
-binary_logistic_head/metrics/auc/LogicalAnd_3
LogicalAnd-binary_logistic_head/metrics/auc/LogicalNot_1(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_4Cast-binary_logistic_head/metrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_3Sum*binary_logistic_head/metrics/auc/ToFloat_48binary_logistic_head/metrics/auc/Sum_3/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
С
,binary_logistic_head/metrics/auc/AssignAdd_3	AssignAdd0binary_logistic_head/metrics/auc/false_positives&binary_logistic_head/metrics/auc/Sum_3*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
k
&binary_logistic_head/metrics/auc/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
п
$binary_logistic_head/metrics/auc/addAdd4binary_logistic_head/metrics/auc/true_positives/read&binary_logistic_head/metrics/auc/add/y*
T0*
_output_shapes	
:╚
└
&binary_logistic_head/metrics/auc/add_1Add4binary_logistic_head/metrics/auc/true_positives/read5binary_logistic_head/metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_2Add&binary_logistic_head/metrics/auc/add_1(binary_logistic_head/metrics/auc/add_2/y*
T0*
_output_shapes	
:╚
г
$binary_logistic_head/metrics/auc/divRealDiv$binary_logistic_head/metrics/auc/add&binary_logistic_head/metrics/auc/add_2*
T0*
_output_shapes	
:╚
└
&binary_logistic_head/metrics/auc/add_3Add5binary_logistic_head/metrics/auc/false_positives/read4binary_logistic_head/metrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_4/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_4Add&binary_logistic_head/metrics/auc/add_3(binary_logistic_head/metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
╢
&binary_logistic_head/metrics/auc/div_1RealDiv5binary_logistic_head/metrics/auc/false_positives/read&binary_logistic_head/metrics/auc/add_4*
T0*
_output_shapes	
:╚
А
6binary_logistic_head/metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_1StridedSlice&binary_logistic_head/metrics/auc/div_16binary_logistic_head/metrics/auc/strided_slice_1/stack8binary_logistic_head/metrics/auc/strided_slice_1/stack_18binary_logistic_head/metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_2StridedSlice&binary_logistic_head/metrics/auc/div_16binary_logistic_head/metrics/auc/strided_slice_2/stack8binary_logistic_head/metrics/auc/strided_slice_2/stack_18binary_logistic_head/metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╡
$binary_logistic_head/metrics/auc/subSub0binary_logistic_head/metrics/auc/strided_slice_10binary_logistic_head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
А
6binary_logistic_head/metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
й
0binary_logistic_head/metrics/auc/strided_slice_3StridedSlice$binary_logistic_head/metrics/auc/div6binary_logistic_head/metrics/auc/strided_slice_3/stack8binary_logistic_head/metrics/auc/strided_slice_3/stack_18binary_logistic_head/metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
й
0binary_logistic_head/metrics/auc/strided_slice_4StridedSlice$binary_logistic_head/metrics/auc/div6binary_logistic_head/metrics/auc/strided_slice_4/stack8binary_logistic_head/metrics/auc/strided_slice_4/stack_18binary_logistic_head/metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╖
&binary_logistic_head/metrics/auc/add_5Add0binary_logistic_head/metrics/auc/strided_slice_30binary_logistic_head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
o
*binary_logistic_head/metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
н
(binary_logistic_head/metrics/auc/truedivRealDiv&binary_logistic_head/metrics/auc/add_5*binary_logistic_head/metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
б
$binary_logistic_head/metrics/auc/MulMul$binary_logistic_head/metrics/auc/sub(binary_logistic_head/metrics/auc/truediv*
T0*
_output_shapes	
:╟
r
(binary_logistic_head/metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
╗
&binary_logistic_head/metrics/auc/valueSum$binary_logistic_head/metrics/auc/Mul(binary_logistic_head/metrics/auc/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
m
(binary_logistic_head/metrics/auc/add_6/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
й
&binary_logistic_head/metrics/auc/add_6Add*binary_logistic_head/metrics/auc/AssignAdd(binary_logistic_head/metrics/auc/add_6/y*
T0*
_output_shapes	
:╚
н
&binary_logistic_head/metrics/auc/add_7Add*binary_logistic_head/metrics/auc/AssignAdd,binary_logistic_head/metrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_8/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_8Add&binary_logistic_head/metrics/auc/add_7(binary_logistic_head/metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
з
&binary_logistic_head/metrics/auc/div_2RealDiv&binary_logistic_head/metrics/auc/add_6&binary_logistic_head/metrics/auc/add_8*
T0*
_output_shapes	
:╚
п
&binary_logistic_head/metrics/auc/add_9Add,binary_logistic_head/metrics/auc/AssignAdd_3,binary_logistic_head/metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
n
)binary_logistic_head/metrics/auc/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
з
'binary_logistic_head/metrics/auc/add_10Add&binary_logistic_head/metrics/auc/add_9)binary_logistic_head/metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
о
&binary_logistic_head/metrics/auc/div_3RealDiv,binary_logistic_head/metrics/auc/AssignAdd_3'binary_logistic_head/metrics/auc/add_10*
T0*
_output_shapes	
:╚
А
6binary_logistic_head/metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_5StridedSlice&binary_logistic_head/metrics/auc/div_36binary_logistic_head/metrics/auc/strided_slice_5/stack8binary_logistic_head/metrics/auc/strided_slice_5/stack_18binary_logistic_head/metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_6StridedSlice&binary_logistic_head/metrics/auc/div_36binary_logistic_head/metrics/auc/strided_slice_6/stack8binary_logistic_head/metrics/auc/strided_slice_6/stack_18binary_logistic_head/metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╖
&binary_logistic_head/metrics/auc/sub_1Sub0binary_logistic_head/metrics/auc/strided_slice_50binary_logistic_head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
А
6binary_logistic_head/metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_7StridedSlice&binary_logistic_head/metrics/auc/div_26binary_logistic_head/metrics/auc/strided_slice_7/stack8binary_logistic_head/metrics/auc/strided_slice_7/stack_18binary_logistic_head/metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_8StridedSlice&binary_logistic_head/metrics/auc/div_26binary_logistic_head/metrics/auc/strided_slice_8/stack8binary_logistic_head/metrics/auc/strided_slice_8/stack_18binary_logistic_head/metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╕
'binary_logistic_head/metrics/auc/add_11Add0binary_logistic_head/metrics/auc/strided_slice_70binary_logistic_head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
q
,binary_logistic_head/metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
▓
*binary_logistic_head/metrics/auc/truediv_1RealDiv'binary_logistic_head/metrics/auc/add_11,binary_logistic_head/metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
з
&binary_logistic_head/metrics/auc/Mul_1Mul&binary_logistic_head/metrics/auc/sub_1*binary_logistic_head/metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
r
(binary_logistic_head/metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
┴
*binary_logistic_head/metrics/auc/update_opSum&binary_logistic_head/metrics/auc/Mul_1(binary_logistic_head/metrics/auc/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

#binary_logistic_head/metrics/Cast_1Casthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
Б
0binary_logistic_head/metrics/auc_1/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╥
*binary_logistic_head/metrics/auc_1/ReshapeReshape)binary_logistic_head/predictions/logistic0binary_logistic_head/metrics/auc_1/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
Г
2binary_logistic_head/metrics/auc_1/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╨
,binary_logistic_head/metrics/auc_1/Reshape_1Reshape#binary_logistic_head/metrics/Cast_12binary_logistic_head/metrics/auc_1/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
Т
(binary_logistic_head/metrics/auc_1/ShapeShape*binary_logistic_head/metrics/auc_1/Reshape*
out_type0*
T0*
_output_shapes
:
А
6binary_logistic_head/metrics/auc_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
и
0binary_logistic_head/metrics/auc_1/strided_sliceStridedSlice(binary_logistic_head/metrics/auc_1/Shape6binary_logistic_head/metrics/auc_1/strided_slice/stack8binary_logistic_head/metrics/auc_1/strided_slice/stack_18binary_logistic_head/metrics/auc_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Ч
(binary_logistic_head/metrics/auc_1/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
{
1binary_logistic_head/metrics/auc_1/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
╬
-binary_logistic_head/metrics/auc_1/ExpandDims
ExpandDims(binary_logistic_head/metrics/auc_1/Const1binary_logistic_head/metrics/auc_1/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
l
*binary_logistic_head/metrics/auc_1/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╚
(binary_logistic_head/metrics/auc_1/stackPack*binary_logistic_head/metrics/auc_1/stack/00binary_logistic_head/metrics/auc_1/strided_slice*
N*
T0*
_output_shapes
:*

axis 
═
'binary_logistic_head/metrics/auc_1/TileTile-binary_logistic_head/metrics/auc_1/ExpandDims(binary_logistic_head/metrics/auc_1/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
Ж
1binary_logistic_head/metrics/auc_1/transpose/RankRank*binary_logistic_head/metrics/auc_1/Reshape*
T0*
_output_shapes
: 
t
2binary_logistic_head/metrics/auc_1/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
┐
0binary_logistic_head/metrics/auc_1/transpose/subSub1binary_logistic_head/metrics/auc_1/transpose/Rank2binary_logistic_head/metrics/auc_1/transpose/sub/y*
T0*
_output_shapes
: 
z
8binary_logistic_head/metrics/auc_1/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
z
8binary_logistic_head/metrics/auc_1/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
К
2binary_logistic_head/metrics/auc_1/transpose/RangeRange8binary_logistic_head/metrics/auc_1/transpose/Range/start1binary_logistic_head/metrics/auc_1/transpose/Rank8binary_logistic_head/metrics/auc_1/transpose/Range/delta*

Tidx0*
_output_shapes
:
─
2binary_logistic_head/metrics/auc_1/transpose/sub_1Sub0binary_logistic_head/metrics/auc_1/transpose/sub2binary_logistic_head/metrics/auc_1/transpose/Range*
T0*
_output_shapes
:
╪
,binary_logistic_head/metrics/auc_1/transpose	Transpose*binary_logistic_head/metrics/auc_1/Reshape2binary_logistic_head/metrics/auc_1/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
Д
3binary_logistic_head/metrics/auc_1/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
┘
)binary_logistic_head/metrics/auc_1/Tile_1Tile,binary_logistic_head/metrics/auc_1/transpose3binary_logistic_head/metrics/auc_1/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
╝
*binary_logistic_head/metrics/auc_1/GreaterGreater)binary_logistic_head/metrics/auc_1/Tile_1'binary_logistic_head/metrics/auc_1/Tile*
T0*(
_output_shapes
:╚         
С
-binary_logistic_head/metrics/auc_1/LogicalNot
LogicalNot*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
Д
3binary_logistic_head/metrics/auc_1/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
┘
)binary_logistic_head/metrics/auc_1/Tile_2Tile,binary_logistic_head/metrics/auc_1/Reshape_13binary_logistic_head/metrics/auc_1/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
Т
/binary_logistic_head/metrics/auc_1/LogicalNot_1
LogicalNot)binary_logistic_head/metrics/auc_1/Tile_2*(
_output_shapes
:╚         
w
(binary_logistic_head/metrics/auc_1/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Я
1binary_logistic_head/metrics/auc_1/true_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
┤
8binary_logistic_head/metrics/auc_1/true_positives/AssignAssign1binary_logistic_head/metrics/auc_1/true_positives(binary_logistic_head/metrics/auc_1/zeros*
validate_shape(*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
с
6binary_logistic_head/metrics/auc_1/true_positives/readIdentity1binary_logistic_head/metrics/auc_1/true_positives*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
T0*
_output_shapes	
:╚
╝
-binary_logistic_head/metrics/auc_1/LogicalAnd
LogicalAnd)binary_logistic_head/metrics/auc_1/Tile_2*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
е
,binary_logistic_head/metrics/auc_1/ToFloat_1Cast-binary_logistic_head/metrics/auc_1/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc_1/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╪
&binary_logistic_head/metrics/auc_1/SumSum,binary_logistic_head/metrics/auc_1/ToFloat_18binary_logistic_head/metrics/auc_1/Sum/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
У
,binary_logistic_head/metrics/auc_1/AssignAdd	AssignAdd1binary_logistic_head/metrics/auc_1/true_positives&binary_logistic_head/metrics/auc_1/Sum*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
а
2binary_logistic_head/metrics/auc_1/false_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
╣
9binary_logistic_head/metrics/auc_1/false_negatives/AssignAssign2binary_logistic_head/metrics/auc_1/false_negatives*binary_logistic_head/metrics/auc_1/zeros_1*
validate_shape(*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
ф
7binary_logistic_head/metrics/auc_1/false_negatives/readIdentity2binary_logistic_head/metrics/auc_1/false_negatives*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
T0*
_output_shapes	
:╚
┴
/binary_logistic_head/metrics/auc_1/LogicalAnd_1
LogicalAnd)binary_logistic_head/metrics/auc_1/Tile_2-binary_logistic_head/metrics/auc_1/LogicalNot*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_2Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_1Sum,binary_logistic_head/metrics/auc_1/ToFloat_2:binary_logistic_head/metrics/auc_1/Sum_1/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
Щ
.binary_logistic_head/metrics/auc_1/AssignAdd_1	AssignAdd2binary_logistic_head/metrics/auc_1/false_negatives(binary_logistic_head/metrics/auc_1/Sum_1*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Я
1binary_logistic_head/metrics/auc_1/true_negatives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
╢
8binary_logistic_head/metrics/auc_1/true_negatives/AssignAssign1binary_logistic_head/metrics/auc_1/true_negatives*binary_logistic_head/metrics/auc_1/zeros_2*
validate_shape(*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
с
6binary_logistic_head/metrics/auc_1/true_negatives/readIdentity1binary_logistic_head/metrics/auc_1/true_negatives*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
T0*
_output_shapes	
:╚
╟
/binary_logistic_head/metrics/auc_1/LogicalAnd_2
LogicalAnd/binary_logistic_head/metrics/auc_1/LogicalNot_1-binary_logistic_head/metrics/auc_1/LogicalNot*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_3Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_2Sum,binary_logistic_head/metrics/auc_1/ToFloat_3:binary_logistic_head/metrics/auc_1/Sum_2/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
Ч
.binary_logistic_head/metrics/auc_1/AssignAdd_2	AssignAdd1binary_logistic_head/metrics/auc_1/true_negatives(binary_logistic_head/metrics/auc_1/Sum_2*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
а
2binary_logistic_head/metrics/auc_1/false_positives
VariableV2*
dtype0*
shape:╚*
	container *
shared_name *
_output_shapes	
:╚
╣
9binary_logistic_head/metrics/auc_1/false_positives/AssignAssign2binary_logistic_head/metrics/auc_1/false_positives*binary_logistic_head/metrics/auc_1/zeros_3*
validate_shape(*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
ф
7binary_logistic_head/metrics/auc_1/false_positives/readIdentity2binary_logistic_head/metrics/auc_1/false_positives*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
T0*
_output_shapes	
:╚
─
/binary_logistic_head/metrics/auc_1/LogicalAnd_3
LogicalAnd/binary_logistic_head/metrics/auc_1/LogicalNot_1*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_4Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_3Sum,binary_logistic_head/metrics/auc_1/ToFloat_4:binary_logistic_head/metrics/auc_1/Sum_3/reduction_indices*
_output_shapes	
:╚*
T0*
	keep_dims( *

Tidx0
Щ
.binary_logistic_head/metrics/auc_1/AssignAdd_3	AssignAdd2binary_logistic_head/metrics/auc_1/false_positives(binary_logistic_head/metrics/auc_1/Sum_3*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc_1/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
╡
&binary_logistic_head/metrics/auc_1/addAdd6binary_logistic_head/metrics/auc_1/true_positives/read(binary_logistic_head/metrics/auc_1/add/y*
T0*
_output_shapes	
:╚
╞
(binary_logistic_head/metrics/auc_1/add_1Add6binary_logistic_head/metrics/auc_1/true_positives/read7binary_logistic_head/metrics/auc_1/false_negatives/read*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_2Add(binary_logistic_head/metrics/auc_1/add_1*binary_logistic_head/metrics/auc_1/add_2/y*
T0*
_output_shapes	
:╚
й
&binary_logistic_head/metrics/auc_1/divRealDiv&binary_logistic_head/metrics/auc_1/add(binary_logistic_head/metrics/auc_1/add_2*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_3/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
╣
(binary_logistic_head/metrics/auc_1/add_3Add6binary_logistic_head/metrics/auc_1/true_positives/read*binary_logistic_head/metrics/auc_1/add_3/y*
T0*
_output_shapes	
:╚
╞
(binary_logistic_head/metrics/auc_1/add_4Add6binary_logistic_head/metrics/auc_1/true_positives/read7binary_logistic_head/metrics/auc_1/false_positives/read*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_5/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_5Add(binary_logistic_head/metrics/auc_1/add_4*binary_logistic_head/metrics/auc_1/add_5/y*
T0*
_output_shapes	
:╚
н
(binary_logistic_head/metrics/auc_1/div_1RealDiv(binary_logistic_head/metrics/auc_1/add_3(binary_logistic_head/metrics/auc_1/add_5*
T0*
_output_shapes	
:╚
В
8binary_logistic_head/metrics/auc_1/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
│
2binary_logistic_head/metrics/auc_1/strided_slice_1StridedSlice&binary_logistic_head/metrics/auc_1/div8binary_logistic_head/metrics/auc_1/strided_slice_1/stack:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
│
2binary_logistic_head/metrics/auc_1/strided_slice_2StridedSlice&binary_logistic_head/metrics/auc_1/div8binary_logistic_head/metrics/auc_1/strided_slice_2/stack:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╗
&binary_logistic_head/metrics/auc_1/subSub2binary_logistic_head/metrics/auc_1/strided_slice_12binary_logistic_head/metrics/auc_1/strided_slice_2*
T0*
_output_shapes	
:╟
В
8binary_logistic_head/metrics/auc_1/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_3StridedSlice(binary_logistic_head/metrics/auc_1/div_18binary_logistic_head/metrics/auc_1/strided_slice_3/stack:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_4StridedSlice(binary_logistic_head/metrics/auc_1/div_18binary_logistic_head/metrics/auc_1/strided_slice_4/stack:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╜
(binary_logistic_head/metrics/auc_1/add_6Add2binary_logistic_head/metrics/auc_1/strided_slice_32binary_logistic_head/metrics/auc_1/strided_slice_4*
T0*
_output_shapes	
:╟
q
,binary_logistic_head/metrics/auc_1/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
│
*binary_logistic_head/metrics/auc_1/truedivRealDiv(binary_logistic_head/metrics/auc_1/add_6,binary_logistic_head/metrics/auc_1/truediv/y*
T0*
_output_shapes	
:╟
з
&binary_logistic_head/metrics/auc_1/MulMul&binary_logistic_head/metrics/auc_1/sub*binary_logistic_head/metrics/auc_1/truediv*
T0*
_output_shapes	
:╟
t
*binary_logistic_head/metrics/auc_1/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
┴
(binary_logistic_head/metrics/auc_1/valueSum&binary_logistic_head/metrics/auc_1/Mul*binary_logistic_head/metrics/auc_1/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
o
*binary_logistic_head/metrics/auc_1/add_7/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
п
(binary_logistic_head/metrics/auc_1/add_7Add,binary_logistic_head/metrics/auc_1/AssignAdd*binary_logistic_head/metrics/auc_1/add_7/y*
T0*
_output_shapes	
:╚
│
(binary_logistic_head/metrics/auc_1/add_8Add,binary_logistic_head/metrics/auc_1/AssignAdd.binary_logistic_head/metrics/auc_1/AssignAdd_1*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_9/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_9Add(binary_logistic_head/metrics/auc_1/add_8*binary_logistic_head/metrics/auc_1/add_9/y*
T0*
_output_shapes	
:╚
н
(binary_logistic_head/metrics/auc_1/div_2RealDiv(binary_logistic_head/metrics/auc_1/add_7(binary_logistic_head/metrics/auc_1/add_9*
T0*
_output_shapes	
:╚
p
+binary_logistic_head/metrics/auc_1/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
▒
)binary_logistic_head/metrics/auc_1/add_10Add,binary_logistic_head/metrics/auc_1/AssignAdd+binary_logistic_head/metrics/auc_1/add_10/y*
T0*
_output_shapes	
:╚
┤
)binary_logistic_head/metrics/auc_1/add_11Add,binary_logistic_head/metrics/auc_1/AssignAdd.binary_logistic_head/metrics/auc_1/AssignAdd_3*
T0*
_output_shapes	
:╚
p
+binary_logistic_head/metrics/auc_1/add_12/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
о
)binary_logistic_head/metrics/auc_1/add_12Add)binary_logistic_head/metrics/auc_1/add_11+binary_logistic_head/metrics/auc_1/add_12/y*
T0*
_output_shapes	
:╚
п
(binary_logistic_head/metrics/auc_1/div_3RealDiv)binary_logistic_head/metrics/auc_1/add_10)binary_logistic_head/metrics/auc_1/add_12*
T0*
_output_shapes	
:╚
В
8binary_logistic_head/metrics/auc_1/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_5StridedSlice(binary_logistic_head/metrics/auc_1/div_28binary_logistic_head/metrics/auc_1/strided_slice_5/stack:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_6StridedSlice(binary_logistic_head/metrics/auc_1/div_28binary_logistic_head/metrics/auc_1/strided_slice_6/stack:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╜
(binary_logistic_head/metrics/auc_1/sub_1Sub2binary_logistic_head/metrics/auc_1/strided_slice_52binary_logistic_head/metrics/auc_1/strided_slice_6*
T0*
_output_shapes	
:╟
В
8binary_logistic_head/metrics/auc_1/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_7StridedSlice(binary_logistic_head/metrics/auc_1/div_38binary_logistic_head/metrics/auc_1/strided_slice_7/stack:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_8StridedSlice(binary_logistic_head/metrics/auc_1/div_38binary_logistic_head/metrics/auc_1/strided_slice_8/stack:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╛
)binary_logistic_head/metrics/auc_1/add_13Add2binary_logistic_head/metrics/auc_1/strided_slice_72binary_logistic_head/metrics/auc_1/strided_slice_8*
T0*
_output_shapes	
:╟
s
.binary_logistic_head/metrics/auc_1/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
╕
,binary_logistic_head/metrics/auc_1/truediv_1RealDiv)binary_logistic_head/metrics/auc_1/add_13.binary_logistic_head/metrics/auc_1/truediv_1/y*
T0*
_output_shapes	
:╟
н
(binary_logistic_head/metrics/auc_1/Mul_1Mul(binary_logistic_head/metrics/auc_1/sub_1,binary_logistic_head/metrics/auc_1/truediv_1*
T0*
_output_shapes	
:╟
t
*binary_logistic_head/metrics/auc_1/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
╟
,binary_logistic_head/metrics/auc_1/update_opSum(binary_logistic_head/metrics/auc_1/Mul_1*binary_logistic_head/metrics/auc_1/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
p
+binary_logistic_head/metrics/GreaterEqual/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
├
)binary_logistic_head/metrics/GreaterEqualGreaterEqual)binary_logistic_head/predictions/logistic+binary_logistic_head/metrics/GreaterEqual/y*
T0*'
_output_shapes
:         
Ъ
&binary_logistic_head/metrics/ToFloat_6Cast)binary_logistic_head/metrics/GreaterEqual*

DstT0*

SrcT0
*'
_output_shapes
:         
Ф
#binary_logistic_head/metrics/Cast_2Cast&binary_logistic_head/metrics/ToFloat_6*

DstT0	*

SrcT0*'
_output_shapes
:         
Ч
$binary_logistic_head/metrics/Equal_1Equal#binary_logistic_head/metrics/Cast_2hash_table_Lookup*
T0	*'
_output_shapes
:         
Х
&binary_logistic_head/metrics/ToFloat_7Cast$binary_logistic_head/metrics/Equal_1*

DstT0*

SrcT0
*'
_output_shapes
:         
r
-binary_logistic_head/metrics/accuracy_1/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
С
-binary_logistic_head/metrics/accuracy_1/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
и
4binary_logistic_head/metrics/accuracy_1/total/AssignAssign-binary_logistic_head/metrics/accuracy_1/total-binary_logistic_head/metrics/accuracy_1/zeros*
validate_shape(*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
use_locking(*
T0*
_output_shapes
: 
╨
2binary_logistic_head/metrics/accuracy_1/total/readIdentity-binary_logistic_head/metrics/accuracy_1/total*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy_1/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
С
-binary_logistic_head/metrics/accuracy_1/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
к
4binary_logistic_head/metrics/accuracy_1/count/AssignAssign-binary_logistic_head/metrics/accuracy_1/count/binary_logistic_head/metrics/accuracy_1/zeros_1*
validate_shape(*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
use_locking(*
T0*
_output_shapes
: 
╨
2binary_logistic_head/metrics/accuracy_1/count/readIdentity-binary_logistic_head/metrics/accuracy_1/count*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
T0*
_output_shapes
: 
Н
,binary_logistic_head/metrics/accuracy_1/SizeSize&binary_logistic_head/metrics/ToFloat_7*
out_type0*
T0*
_output_shapes
: 
Ч
1binary_logistic_head/metrics/accuracy_1/ToFloat_1Cast,binary_logistic_head/metrics/accuracy_1/Size*

DstT0*

SrcT0*
_output_shapes
: 
~
-binary_logistic_head/metrics/accuracy_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
╟
+binary_logistic_head/metrics/accuracy_1/SumSum&binary_logistic_head/metrics/ToFloat_7-binary_logistic_head/metrics/accuracy_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Р
1binary_logistic_head/metrics/accuracy_1/AssignAdd	AssignAdd-binary_logistic_head/metrics/accuracy_1/total+binary_logistic_head/metrics/accuracy_1/Sum*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
use_locking( *
T0*
_output_shapes
: 
┴
3binary_logistic_head/metrics/accuracy_1/AssignAdd_1	AssignAdd-binary_logistic_head/metrics/accuracy_1/count1binary_logistic_head/metrics/accuracy_1/ToFloat_1'^binary_logistic_head/metrics/ToFloat_7*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
use_locking( *
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy_1/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┬
/binary_logistic_head/metrics/accuracy_1/GreaterGreater2binary_logistic_head/metrics/accuracy_1/count/read1binary_logistic_head/metrics/accuracy_1/Greater/y*
T0*
_output_shapes
: 
├
/binary_logistic_head/metrics/accuracy_1/truedivRealDiv2binary_logistic_head/metrics/accuracy_1/total/read2binary_logistic_head/metrics/accuracy_1/count/read*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy_1/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ы
-binary_logistic_head/metrics/accuracy_1/valueSelect/binary_logistic_head/metrics/accuracy_1/Greater/binary_logistic_head/metrics/accuracy_1/truediv/binary_logistic_head/metrics/accuracy_1/value/e*
T0*
_output_shapes
: 
x
3binary_logistic_head/metrics/accuracy_1/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╟
1binary_logistic_head/metrics/accuracy_1/Greater_1Greater3binary_logistic_head/metrics/accuracy_1/AssignAdd_13binary_logistic_head/metrics/accuracy_1/Greater_1/y*
T0*
_output_shapes
: 
┼
1binary_logistic_head/metrics/accuracy_1/truediv_1RealDiv1binary_logistic_head/metrics/accuracy_1/AssignAdd3binary_logistic_head/metrics/accuracy_1/AssignAdd_1*
T0*
_output_shapes
: 
x
3binary_logistic_head/metrics/accuracy_1/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ў
1binary_logistic_head/metrics/accuracy_1/update_opSelect1binary_logistic_head/metrics/accuracy_1/Greater_11binary_logistic_head/metrics/accuracy_1/truediv_13binary_logistic_head/metrics/accuracy_1/update_op/e*
T0*
_output_shapes
: 
Х
9binary_logistic_head/metrics/precision_at_thresholds/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/precision_at_thresholds/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ў
<binary_logistic_head/metrics/precision_at_thresholds/ReshapeReshape)binary_logistic_head/predictions/logisticBbinary_logistic_head/metrics/precision_at_thresholds/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
Х
Dbinary_logistic_head/metrics/precision_at_thresholds/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
К
>binary_logistic_head/metrics/precision_at_thresholds/Reshape_1Reshape9binary_logistic_head/metrics/precision_at_thresholds/CastDbinary_logistic_head/metrics/precision_at_thresholds/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
╢
:binary_logistic_head/metrics/precision_at_thresholds/ShapeShape<binary_logistic_head/metrics/precision_at_thresholds/Reshape*
out_type0*
T0*
_output_shapes
:
Т
Hbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ф
Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ф
Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
В
Bbinary_logistic_head/metrics/precision_at_thresholds/strided_sliceStridedSlice:binary_logistic_head/metrics/precision_at_thresholds/ShapeHbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stackJbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_1Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
З
:binary_logistic_head/metrics/precision_at_thresholds/ConstConst*
dtype0*
valueB*   ?*
_output_shapes
:
Н
Cbinary_logistic_head/metrics/precision_at_thresholds/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Г
?binary_logistic_head/metrics/precision_at_thresholds/ExpandDims
ExpandDims:binary_logistic_head/metrics/precision_at_thresholds/ConstCbinary_logistic_head/metrics/precision_at_thresholds/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
~
<binary_logistic_head/metrics/precision_at_thresholds/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
■
:binary_logistic_head/metrics/precision_at_thresholds/stackPack<binary_logistic_head/metrics/precision_at_thresholds/stack/0Bbinary_logistic_head/metrics/precision_at_thresholds/strided_slice*
N*
T0*
_output_shapes
:*

axis 
В
9binary_logistic_head/metrics/precision_at_thresholds/TileTile?binary_logistic_head/metrics/precision_at_thresholds/ExpandDims:binary_logistic_head/metrics/precision_at_thresholds/stack*

Tmultiples0*
T0*'
_output_shapes
:         
к
Cbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankRank<binary_logistic_head/metrics/precision_at_thresholds/Reshape*
T0*
_output_shapes
: 
Ж
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
ї
Bbinary_logistic_head/metrics/precision_at_thresholds/transpose/subSubCbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankDbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub/y*
T0*
_output_shapes
: 
М
Jbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
М
Jbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╥
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/RangeRangeJbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/startCbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankJbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/delta*

Tidx0*
_output_shapes
:
·
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub_1SubBbinary_logistic_head/metrics/precision_at_thresholds/transpose/subDbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range*
T0*
_output_shapes
:
О
>binary_logistic_head/metrics/precision_at_thresholds/transpose	Transpose<binary_logistic_head/metrics/precision_at_thresholds/ReshapeDbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
Ц
Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_1/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
О
;binary_logistic_head/metrics/precision_at_thresholds/Tile_1Tile>binary_logistic_head/metrics/precision_at_thresholds/transposeEbinary_logistic_head/metrics/precision_at_thresholds/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
ё
<binary_logistic_head/metrics/precision_at_thresholds/GreaterGreater;binary_logistic_head/metrics/precision_at_thresholds/Tile_19binary_logistic_head/metrics/precision_at_thresholds/Tile*
T0*'
_output_shapes
:         
Ц
Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_2/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
О
;binary_logistic_head/metrics/precision_at_thresholds/Tile_2Tile>binary_logistic_head/metrics/precision_at_thresholds/Reshape_1Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_2/multiples*

Tmultiples0*
T0
*'
_output_shapes
:         
│
?binary_logistic_head/metrics/precision_at_thresholds/LogicalNot
LogicalNot;binary_logistic_head/metrics/precision_at_thresholds/Tile_2*'
_output_shapes
:         
З
:binary_logistic_head/metrics/precision_at_thresholds/zerosConst*
dtype0*
valueB*    *
_output_shapes
:
п
Cbinary_logistic_head/metrics/precision_at_thresholds/true_positives
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
√
Jbinary_logistic_head/metrics/precision_at_thresholds/true_positives/AssignAssignCbinary_logistic_head/metrics/precision_at_thresholds/true_positives:binary_logistic_head/metrics/precision_at_thresholds/zeros*
validate_shape(*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
use_locking(*
T0*
_output_shapes
:
Ц
Hbinary_logistic_head/metrics/precision_at_thresholds/true_positives/readIdentityCbinary_logistic_head/metrics/precision_at_thresholds/true_positives*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
T0*
_output_shapes
:
ё
?binary_logistic_head/metrics/precision_at_thresholds/LogicalAnd
LogicalAnd;binary_logistic_head/metrics/precision_at_thresholds/Tile_2<binary_logistic_head/metrics/precision_at_thresholds/Greater*'
_output_shapes
:         
╚
>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_1Cast?binary_logistic_head/metrics/precision_at_thresholds/LogicalAnd*

DstT0*

SrcT0
*'
_output_shapes
:         
М
Jbinary_logistic_head/metrics/precision_at_thresholds/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Н
8binary_logistic_head/metrics/precision_at_thresholds/SumSum>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_1Jbinary_logistic_head/metrics/precision_at_thresholds/Sum/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
┌
>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd	AssignAddCbinary_logistic_head/metrics/precision_at_thresholds/true_positives8binary_logistic_head/metrics/precision_at_thresholds/Sum*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
use_locking( *
T0*
_output_shapes
:
Й
<binary_logistic_head/metrics/precision_at_thresholds/zeros_1Const*
dtype0*
valueB*    *
_output_shapes
:
░
Dbinary_logistic_head/metrics/precision_at_thresholds/false_positives
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
А
Kbinary_logistic_head/metrics/precision_at_thresholds/false_positives/AssignAssignDbinary_logistic_head/metrics/precision_at_thresholds/false_positives<binary_logistic_head/metrics/precision_at_thresholds/zeros_1*
validate_shape(*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
use_locking(*
T0*
_output_shapes
:
Щ
Ibinary_logistic_head/metrics/precision_at_thresholds/false_positives/readIdentityDbinary_logistic_head/metrics/precision_at_thresholds/false_positives*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
T0*
_output_shapes
:
ў
Abinary_logistic_head/metrics/precision_at_thresholds/LogicalAnd_1
LogicalAnd?binary_logistic_head/metrics/precision_at_thresholds/LogicalNot<binary_logistic_head/metrics/precision_at_thresholds/Greater*'
_output_shapes
:         
╩
>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_2CastAbinary_logistic_head/metrics/precision_at_thresholds/LogicalAnd_1*

DstT0*

SrcT0
*'
_output_shapes
:         
О
Lbinary_logistic_head/metrics/precision_at_thresholds/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
С
:binary_logistic_head/metrics/precision_at_thresholds/Sum_1Sum>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_2Lbinary_logistic_head/metrics/precision_at_thresholds/Sum_1/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
р
@binary_logistic_head/metrics/precision_at_thresholds/AssignAdd_1	AssignAddDbinary_logistic_head/metrics/precision_at_thresholds/false_positives:binary_logistic_head/metrics/precision_at_thresholds/Sum_1*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
use_locking( *
T0*
_output_shapes
:

:binary_logistic_head/metrics/precision_at_thresholds/add/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
ъ
8binary_logistic_head/metrics/precision_at_thresholds/addAdd:binary_logistic_head/metrics/precision_at_thresholds/add/xHbinary_logistic_head/metrics/precision_at_thresholds/true_positives/read*
T0*
_output_shapes
:
ы
:binary_logistic_head/metrics/precision_at_thresholds/add_1Add8binary_logistic_head/metrics/precision_at_thresholds/addIbinary_logistic_head/metrics/precision_at_thresholds/false_positives/read*
T0*
_output_shapes
:
·
Dbinary_logistic_head/metrics/precision_at_thresholds/precision_valueRealDivHbinary_logistic_head/metrics/precision_at_thresholds/true_positives/read:binary_logistic_head/metrics/precision_at_thresholds/add_1*
T0*
_output_shapes
:
Б
<binary_logistic_head/metrics/precision_at_thresholds/add_2/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
ф
:binary_logistic_head/metrics/precision_at_thresholds/add_2Add<binary_logistic_head/metrics/precision_at_thresholds/add_2/x>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd*
T0*
_output_shapes
:
ф
:binary_logistic_head/metrics/precision_at_thresholds/add_3Add:binary_logistic_head/metrics/precision_at_thresholds/add_2@binary_logistic_head/metrics/precision_at_thresholds/AssignAdd_1*
T0*
_output_shapes
:
Ї
Hbinary_logistic_head/metrics/precision_at_thresholds/precision_update_opRealDiv>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd:binary_logistic_head/metrics/precision_at_thresholds/add_3*
T0*
_output_shapes
:
к
$binary_logistic_head/metrics/SqueezeSqueezeDbinary_logistic_head/metrics/precision_at_thresholds/precision_value*
squeeze_dims
 *
T0*
_output_shapes
: 
░
&binary_logistic_head/metrics/Squeeze_1SqueezeHbinary_logistic_head/metrics/precision_at_thresholds/precision_update_op*
squeeze_dims
 *
T0*
_output_shapes
: 
Т
6binary_logistic_head/metrics/recall_at_thresholds/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
Р
?binary_logistic_head/metrics/recall_at_thresholds/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ё
9binary_logistic_head/metrics/recall_at_thresholds/ReshapeReshape)binary_logistic_head/predictions/logistic?binary_logistic_head/metrics/recall_at_thresholds/Reshape/shape*'
_output_shapes
:         *
T0*
Tshape0
Т
Abinary_logistic_head/metrics/recall_at_thresholds/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Б
;binary_logistic_head/metrics/recall_at_thresholds/Reshape_1Reshape6binary_logistic_head/metrics/recall_at_thresholds/CastAbinary_logistic_head/metrics/recall_at_thresholds/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
░
7binary_logistic_head/metrics/recall_at_thresholds/ShapeShape9binary_logistic_head/metrics/recall_at_thresholds/Reshape*
out_type0*
T0*
_output_shapes
:
П
Ebinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
С
Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
С
Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
є
?binary_logistic_head/metrics/recall_at_thresholds/strided_sliceStridedSlice7binary_logistic_head/metrics/recall_at_thresholds/ShapeEbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stackGbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_1Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Д
7binary_logistic_head/metrics/recall_at_thresholds/ConstConst*
dtype0*
valueB*   ?*
_output_shapes
:
К
@binary_logistic_head/metrics/recall_at_thresholds/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
·
<binary_logistic_head/metrics/recall_at_thresholds/ExpandDims
ExpandDims7binary_logistic_head/metrics/recall_at_thresholds/Const@binary_logistic_head/metrics/recall_at_thresholds/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
{
9binary_logistic_head/metrics/recall_at_thresholds/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ї
7binary_logistic_head/metrics/recall_at_thresholds/stackPack9binary_logistic_head/metrics/recall_at_thresholds/stack/0?binary_logistic_head/metrics/recall_at_thresholds/strided_slice*
N*
T0*
_output_shapes
:*

axis 
∙
6binary_logistic_head/metrics/recall_at_thresholds/TileTile<binary_logistic_head/metrics/recall_at_thresholds/ExpandDims7binary_logistic_head/metrics/recall_at_thresholds/stack*

Tmultiples0*
T0*'
_output_shapes
:         
д
@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankRank9binary_logistic_head/metrics/recall_at_thresholds/Reshape*
T0*
_output_shapes
: 
Г
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
ь
?binary_logistic_head/metrics/recall_at_thresholds/transpose/subSub@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankAbinary_logistic_head/metrics/recall_at_thresholds/transpose/sub/y*
T0*
_output_shapes
: 
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╞
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/RangeRangeGbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/start@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankGbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/delta*

Tidx0*
_output_shapes
:
ё
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/sub_1Sub?binary_logistic_head/metrics/recall_at_thresholds/transpose/subAbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range*
T0*
_output_shapes
:
Е
;binary_logistic_head/metrics/recall_at_thresholds/transpose	Transpose9binary_logistic_head/metrics/recall_at_thresholds/ReshapeAbinary_logistic_head/metrics/recall_at_thresholds/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_1/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
Е
8binary_logistic_head/metrics/recall_at_thresholds/Tile_1Tile;binary_logistic_head/metrics/recall_at_thresholds/transposeBbinary_logistic_head/metrics/recall_at_thresholds/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
ш
9binary_logistic_head/metrics/recall_at_thresholds/GreaterGreater8binary_logistic_head/metrics/recall_at_thresholds/Tile_16binary_logistic_head/metrics/recall_at_thresholds/Tile*
T0*'
_output_shapes
:         
о
<binary_logistic_head/metrics/recall_at_thresholds/LogicalNot
LogicalNot9binary_logistic_head/metrics/recall_at_thresholds/Greater*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_2/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
Е
8binary_logistic_head/metrics/recall_at_thresholds/Tile_2Tile;binary_logistic_head/metrics/recall_at_thresholds/Reshape_1Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_2/multiples*

Tmultiples0*
T0
*'
_output_shapes
:         
Д
7binary_logistic_head/metrics/recall_at_thresholds/zerosConst*
dtype0*
valueB*    *
_output_shapes
:
м
@binary_logistic_head/metrics/recall_at_thresholds/true_positives
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
я
Gbinary_logistic_head/metrics/recall_at_thresholds/true_positives/AssignAssign@binary_logistic_head/metrics/recall_at_thresholds/true_positives7binary_logistic_head/metrics/recall_at_thresholds/zeros*
validate_shape(*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
use_locking(*
T0*
_output_shapes
:
Н
Ebinary_logistic_head/metrics/recall_at_thresholds/true_positives/readIdentity@binary_logistic_head/metrics/recall_at_thresholds/true_positives*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
T0*
_output_shapes
:
ш
<binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd
LogicalAnd8binary_logistic_head/metrics/recall_at_thresholds/Tile_29binary_logistic_head/metrics/recall_at_thresholds/Greater*'
_output_shapes
:         
┬
;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_1Cast<binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd*

DstT0*

SrcT0
*'
_output_shapes
:         
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Д
5binary_logistic_head/metrics/recall_at_thresholds/SumSum;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_1Gbinary_logistic_head/metrics/recall_at_thresholds/Sum/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
╬
;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd	AssignAdd@binary_logistic_head/metrics/recall_at_thresholds/true_positives5binary_logistic_head/metrics/recall_at_thresholds/Sum*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
use_locking( *
T0*
_output_shapes
:
Ж
9binary_logistic_head/metrics/recall_at_thresholds/zeros_1Const*
dtype0*
valueB*    *
_output_shapes
:
н
Abinary_logistic_head/metrics/recall_at_thresholds/false_negatives
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
Ї
Hbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/AssignAssignAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives9binary_logistic_head/metrics/recall_at_thresholds/zeros_1*
validate_shape(*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
use_locking(*
T0*
_output_shapes
:
Р
Fbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/readIdentityAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
T0*
_output_shapes
:
э
>binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd_1
LogicalAnd8binary_logistic_head/metrics/recall_at_thresholds/Tile_2<binary_logistic_head/metrics/recall_at_thresholds/LogicalNot*'
_output_shapes
:         
─
;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_2Cast>binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd_1*

DstT0*

SrcT0
*'
_output_shapes
:         
Л
Ibinary_logistic_head/metrics/recall_at_thresholds/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
И
7binary_logistic_head/metrics/recall_at_thresholds/Sum_1Sum;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_2Ibinary_logistic_head/metrics/recall_at_thresholds/Sum_1/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
╘
=binary_logistic_head/metrics/recall_at_thresholds/AssignAdd_1	AssignAddAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives7binary_logistic_head/metrics/recall_at_thresholds/Sum_1*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
use_locking( *
T0*
_output_shapes
:
|
7binary_logistic_head/metrics/recall_at_thresholds/add/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
с
5binary_logistic_head/metrics/recall_at_thresholds/addAdd7binary_logistic_head/metrics/recall_at_thresholds/add/xEbinary_logistic_head/metrics/recall_at_thresholds/true_positives/read*
T0*
_output_shapes
:
т
7binary_logistic_head/metrics/recall_at_thresholds/add_1Add5binary_logistic_head/metrics/recall_at_thresholds/addFbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/read*
T0*
_output_shapes
:
ю
>binary_logistic_head/metrics/recall_at_thresholds/recall_valueRealDivEbinary_logistic_head/metrics/recall_at_thresholds/true_positives/read7binary_logistic_head/metrics/recall_at_thresholds/add_1*
T0*
_output_shapes
:
~
9binary_logistic_head/metrics/recall_at_thresholds/add_2/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
█
7binary_logistic_head/metrics/recall_at_thresholds/add_2Add9binary_logistic_head/metrics/recall_at_thresholds/add_2/x;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd*
T0*
_output_shapes
:
█
7binary_logistic_head/metrics/recall_at_thresholds/add_3Add7binary_logistic_head/metrics/recall_at_thresholds/add_2=binary_logistic_head/metrics/recall_at_thresholds/AssignAdd_1*
T0*
_output_shapes
:
ш
Bbinary_logistic_head/metrics/recall_at_thresholds/recall_update_opRealDiv;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd7binary_logistic_head/metrics/recall_at_thresholds/add_3*
T0*
_output_shapes
:
ж
&binary_logistic_head/metrics/Squeeze_2Squeeze>binary_logistic_head/metrics/recall_at_thresholds/recall_value*
squeeze_dims
 *
T0*
_output_shapes
: 
к
&binary_logistic_head/metrics/Squeeze_3SqueezeBbinary_logistic_head/metrics/recall_at_thresholds/recall_update_op*
squeeze_dims
 *
T0*
_output_shapes
: 
м
>binary_logistic_head/_classification_output_alternatives/ShapeShape.binary_logistic_head/predictions/probabilities*
out_type0*
T0*
_output_shapes
:
Ц
Lbinary_logistic_head/_classification_output_alternatives/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ц
Fbinary_logistic_head/_classification_output_alternatives/strided_sliceStridedSlice>binary_logistic_head/_classification_output_alternatives/ShapeLbinary_logistic_head/_classification_output_alternatives/strided_slice/stackNbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_1Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
о
@binary_logistic_head/_classification_output_alternatives/Shape_1Shape.binary_logistic_head/predictions/probabilities*
out_type0*
T0*
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
а
Hbinary_logistic_head/_classification_output_alternatives/strided_slice_1StridedSlice@binary_logistic_head/_classification_output_alternatives/Shape_1Nbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stackPbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_1Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Ж
Dbinary_logistic_head/_classification_output_alternatives/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Ж
Dbinary_logistic_head/_classification_output_alternatives/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╬
>binary_logistic_head/_classification_output_alternatives/rangeRangeDbinary_logistic_head/_classification_output_alternatives/range/startHbinary_logistic_head/_classification_output_alternatives/strided_slice_1Dbinary_logistic_head/_classification_output_alternatives/range/delta*

Tidx0*#
_output_shapes
:         
Й
Gbinary_logistic_head/_classification_output_alternatives/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
Ш
Cbinary_logistic_head/_classification_output_alternatives/ExpandDims
ExpandDims>binary_logistic_head/_classification_output_alternatives/rangeGbinary_logistic_head/_classification_output_alternatives/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
Л
Ibinary_logistic_head/_classification_output_alternatives/Tile/multiples/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ь
Gbinary_logistic_head/_classification_output_alternatives/Tile/multiplesPackFbinary_logistic_head/_classification_output_alternatives/strided_sliceIbinary_logistic_head/_classification_output_alternatives/Tile/multiples/1*
N*
T0*
_output_shapes
:*

axis 
а
=binary_logistic_head/_classification_output_alternatives/TileTileCbinary_logistic_head/_classification_output_alternatives/ExpandDimsGbinary_logistic_head/_classification_output_alternatives/Tile/multiples*

Tmultiples0*
T0*0
_output_shapes
:                  
л
Gbinary_logistic_head/_classification_output_alternatives/classes_tensorAsString=binary_logistic_head/_classification_output_alternatives/Tile*

scientific( *0
_output_shapes
:                  *
	precision         *
width         *
T0*
shortest( *

fill 
Р
$remove_squeezable_dimensions/SqueezeSqueezehash_table_Lookup*
squeeze_dims

         *
T0	*#
_output_shapes
:         
М
EqualEqual(binary_logistic_head/predictions/classes$remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
S
ToFloatCastEqual*

DstT0*

SrcT0
*#
_output_shapes
:         
S
accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
accuracy/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
м
accuracy/total/AssignAssignaccuracy/totalaccuracy/zeros*
validate_shape(*!
_class
loc:@accuracy/total*
use_locking(*
T0*
_output_shapes
: 
s
accuracy/total/readIdentityaccuracy/total*!
_class
loc:@accuracy/total*
T0*
_output_shapes
: 
U
accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
r
accuracy/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
о
accuracy/count/AssignAssignaccuracy/countaccuracy/zeros_1*
validate_shape(*!
_class
loc:@accuracy/count*
use_locking(*
T0*
_output_shapes
: 
s
accuracy/count/readIdentityaccuracy/count*!
_class
loc:@accuracy/count*
T0*
_output_shapes
: 
O
accuracy/SizeSizeToFloat*
out_type0*
T0*
_output_shapes
: 
Y
accuracy/ToFloat_1Castaccuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
X
accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
j
accuracy/SumSumToFloataccuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ф
accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
T0*
_output_shapes
: 
ж
accuracy/AssignAdd_1	AssignAddaccuracy/countaccuracy/ToFloat_1^ToFloat*!
_class
loc:@accuracy/count*
use_locking( *
T0*
_output_shapes
: 
W
accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
e
accuracy/GreaterGreateraccuracy/count/readaccuracy/Greater/y*
T0*
_output_shapes
: 
f
accuracy/truedivRealDivaccuracy/total/readaccuracy/count/read*
T0*
_output_shapes
: 
U
accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
o
accuracy/valueSelectaccuracy/Greateraccuracy/truedivaccuracy/value/e*
T0*
_output_shapes
: 
Y
accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
j
accuracy/Greater_1Greateraccuracy/AssignAdd_1accuracy/Greater_1/y*
T0*
_output_shapes
: 
h
accuracy/truediv_1RealDivaccuracy/AssignAddaccuracy/AssignAdd_1*
T0*
_output_shapes
: 
Y
accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
{
accuracy/update_opSelectaccuracy/Greater_1accuracy/truediv_1accuracy/update_op/e*
T0*
_output_shapes
: 
╟

group_depsNoOp.^binary_logistic_head/metrics/mean_3/update_op,^binary_logistic_head/metrics/mean/update_op+^binary_logistic_head/metrics/auc/update_op2^binary_logistic_head/metrics/accuracy_1/update_op'^binary_logistic_head/metrics/Squeeze_3.^binary_logistic_head/metrics/mean_1/update_op^accuracy/update_op-^binary_logistic_head/metrics/auc_1/update_op'^binary_logistic_head/metrics/Squeeze_1.^binary_logistic_head/metrics/mean_2/update_op
{
eval_step/Initializer/zerosConst*
dtype0	*
_class
loc:@eval_step*
value	B	 R *
_output_shapes
: 
Л
	eval_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@eval_step*
shared_name 
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0	*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0	*
_output_shapes
: 
Q
AssignAdd/valueConst*
dtype0	*
value	B	 R*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0	*
_output_shapes
: 
є
initNoOp^global_step/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign0^linear/linear_model/alpha/weights/part_0/Assign/^linear/linear_model/beta/weights/part_0/Assign/^linear/linear_model/bias_weights/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
╜
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
█
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized(linear/linear_model/alpha/weights/part_0*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized'linear/linear_model/beta/weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized'linear/linear_model/bias_weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized'binary_logistic_head/metrics/mean/total*
dtype0*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized'binary_logistic_head/metrics/mean/count*
dtype0*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
_output_shapes
: 
т
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized+binary_logistic_head/metrics/accuracy/total*
dtype0*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
_output_shapes
: 
т
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized+binary_logistic_head/metrics/accuracy/count*
dtype0*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized)binary_logistic_head/metrics/mean_1/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized)binary_logistic_head/metrics/mean_1/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized)binary_logistic_head/metrics/mean_2/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized)binary_logistic_head/metrics/mean_2/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized)binary_logistic_head/metrics/mean_3/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized)binary_logistic_head/metrics/mean_3/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
_output_shapes
: 
ъ
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized/binary_logistic_head/metrics/auc/true_positives*
dtype0*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
_output_shapes
: 
ь
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized0binary_logistic_head/metrics/auc/false_negatives*
dtype0*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
_output_shapes
: 
ъ
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized/binary_logistic_head/metrics/auc/true_negatives*
dtype0*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
_output_shapes
: 
ь
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized0binary_logistic_head/metrics/auc/false_positives*
dtype0*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
_output_shapes
: 
ю
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized1binary_logistic_head/metrics/auc_1/true_positives*
dtype0*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
_output_shapes
: 
Ё
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized2binary_logistic_head/metrics/auc_1/false_negatives*
dtype0*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
_output_shapes
: 
ю
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized1binary_logistic_head/metrics/auc_1/true_negatives*
dtype0*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
_output_shapes
: 
Ё
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized2binary_logistic_head/metrics/auc_1/false_positives*
dtype0*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
_output_shapes
: 
ц
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized-binary_logistic_head/metrics/accuracy_1/total*
dtype0*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
_output_shapes
: 
ц
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized-binary_logistic_head/metrics/accuracy_1/count*
dtype0*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
_output_shapes
: 
Т
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedCbinary_logistic_head/metrics/precision_at_thresholds/true_positives*
dtype0*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
_output_shapes
: 
Ф
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedDbinary_logistic_head/metrics/precision_at_thresholds/false_positives*
dtype0*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
_output_shapes
: 
М
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitialized@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
dtype0*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives*
dtype0*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_36IsVariableInitializedaccuracy/total*
dtype0*!
_class
loc:@accuracy/total*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_37IsVariableInitializedaccuracy/count*
dtype0*!
_class
loc:@accuracy/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_38IsVariableInitialized	eval_step*
dtype0	*
_class
loc:@eval_step*
_output_shapes
: 
Й
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_357report_uninitialized_variables/IsVariableInitialized_367report_uninitialized_variables/IsVariableInitialized_377report_uninitialized_variables/IsVariableInitialized_38*
N'*
T0
*
_output_shapes
:'*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:'
╨
$report_uninitialized_variables/ConstConst*
dtype0*ў
valueэBъ'Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B(linear/linear_model/alpha/weights/part_0B'linear/linear_model/beta/weights/part_0B'linear/linear_model/bias_weights/part_0B'binary_logistic_head/metrics/mean/totalB'binary_logistic_head/metrics/mean/countB+binary_logistic_head/metrics/accuracy/totalB+binary_logistic_head/metrics/accuracy/countB)binary_logistic_head/metrics/mean_1/totalB)binary_logistic_head/metrics/mean_1/countB)binary_logistic_head/metrics/mean_2/totalB)binary_logistic_head/metrics/mean_2/countB)binary_logistic_head/metrics/mean_3/totalB)binary_logistic_head/metrics/mean_3/countB/binary_logistic_head/metrics/auc/true_positivesB0binary_logistic_head/metrics/auc/false_negativesB/binary_logistic_head/metrics/auc/true_negativesB0binary_logistic_head/metrics/auc/false_positivesB1binary_logistic_head/metrics/auc_1/true_positivesB2binary_logistic_head/metrics/auc_1/false_negativesB1binary_logistic_head/metrics/auc_1/true_negativesB2binary_logistic_head/metrics/auc_1/false_positivesB-binary_logistic_head/metrics/accuracy_1/totalB-binary_logistic_head/metrics/accuracy_1/countBCbinary_logistic_head/metrics/precision_at_thresholds/true_positivesBDbinary_logistic_head/metrics/precision_at_thresholds/false_positivesB@binary_logistic_head/metrics/recall_at_thresholds/true_positivesBAbinary_logistic_head/metrics/recall_at_thresholds/false_negativesBaccuracy/totalBaccuracy/countB	eval_step*
_output_shapes
:'
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:'*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:'*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:'*
T0*
Tshape0
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
_output_shapes
:'*
T0
*
Tshape0
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:         *
T0
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
┐
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
▌
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitialized(linear/linear_model/alpha/weights/part_0*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
_output_shapes
: 
▄
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitialized'linear/linear_model/beta/weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
_output_shapes
: 
▄
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitialized'linear/linear_model/bias_weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: 
а
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_11*
N*
T0
*
_output_shapes
:*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ў
&report_uninitialized_variables_1/ConstConst*
dtype0*Ь
valueТBПBglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B(linear/linear_model/alpha/weights/part_0B'linear/linear_model/beta/weights/part_0B'linear/linear_model/bias_weights/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
┴
init_2NoOp/^binary_logistic_head/metrics/mean/total/Assign/^binary_logistic_head/metrics/mean/count/Assign3^binary_logistic_head/metrics/accuracy/total/Assign3^binary_logistic_head/metrics/accuracy/count/Assign1^binary_logistic_head/metrics/mean_1/total/Assign1^binary_logistic_head/metrics/mean_1/count/Assign1^binary_logistic_head/metrics/mean_2/total/Assign1^binary_logistic_head/metrics/mean_2/count/Assign1^binary_logistic_head/metrics/mean_3/total/Assign1^binary_logistic_head/metrics/mean_3/count/Assign7^binary_logistic_head/metrics/auc/true_positives/Assign8^binary_logistic_head/metrics/auc/false_negatives/Assign7^binary_logistic_head/metrics/auc/true_negatives/Assign8^binary_logistic_head/metrics/auc/false_positives/Assign9^binary_logistic_head/metrics/auc_1/true_positives/Assign:^binary_logistic_head/metrics/auc_1/false_negatives/Assign9^binary_logistic_head/metrics/auc_1/true_negatives/Assign:^binary_logistic_head/metrics/auc_1/false_positives/Assign5^binary_logistic_head/metrics/accuracy_1/total/Assign5^binary_logistic_head/metrics/accuracy_1/count/AssignK^binary_logistic_head/metrics/precision_at_thresholds/true_positives/AssignL^binary_logistic_head/metrics/precision_at_thresholds/false_positives/AssignH^binary_logistic_head/metrics/recall_at_thresholds/true_positives/AssignI^binary_logistic_head/metrics/recall_at_thresholds/false_negatives/Assign^accuracy/total/Assign^accuracy/count/Assign^eval_step/Assign
∙
init_all_tablesNoOp&^string_to_index/hash_table/table_init^^dnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/table_init\^dnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/table_init=^linear/linear_model/alpha/alpha_lookup/hash_table/table_init;^linear/linear_model/beta/beta_lookup/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
Я
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_5000_full-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation-dnn/dnn/hiddenlayer_2/fraction_of_zero_values dnn/dnn/hiddenlayer_2/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation%linear/linear/fraction_of_zero_valueslinear/linear/activation"binary_logistic_head/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_5a457b3906f24157867a6202b2d6f0c4/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ь
save/SaveV2/tensor_namesConst*
dtype0*╧
value┼B┬Bdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_stepB!linear/linear_model/alpha/weightsB linear/linear_model/beta/weightsB linear/linear_model/bias_weights*
_output_shapes
:
ё
save/SaveV2/shape_and_slicesConst*
dtype0*а
valueЦBУB	128 0,128B6 128 0,6:0,128B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32B1 0,1B32 1 0,32:0,1B B2 1 0,2:0,1B2 1 0,2:0,1B1 0,1*
_output_shapes
:
╜
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step-linear/linear_model/alpha/weights/part_0/read,linear/linear_model/beta/weights/part_0/read,linear/linear_model/bias_weights/part_0/read*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
|
save/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
q
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB	128 0,128*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
╔
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes	
:А

save/RestoreV2_1/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*$
valueBB6 128 0,6:0,128*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А
~
save/RestoreV2_2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB64 0,64*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
{
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*&
valueBB128 64 0,128:0,64*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А@
~
save/RestoreV2_4/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_2/biases*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueBB32 0,32*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_4Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_5/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights*
_output_shapes
:
y
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*$
valueBB64 32 0,64:0,32*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:@ 
w
save/RestoreV2_6/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
o
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_6Assigndnn/logits/biases/part_0save/RestoreV2_6*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
x
save/RestoreV2_7/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
w
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*"
valueBB32 1 0,32:0,1*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
─
save/Assign_7Assigndnn/logits/weights/part_0save/RestoreV2_7*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

: 
q
save/RestoreV2_8/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2	*
_output_shapes
:
а
save/Assign_8Assignglobal_stepsave/RestoreV2_8*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
З
save/RestoreV2_9/tensor_namesConst*
dtype0*6
value-B+B!linear/linear_model/alpha/weights*
_output_shapes
:
u
!save/RestoreV2_9/shape_and_slicesConst*
dtype0* 
valueBB2 1 0,2:0,1*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
т
save/Assign_9Assign(linear/linear_model/alpha/weights/part_0save/RestoreV2_9*
validate_shape(*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
use_locking(*
T0*
_output_shapes

:
З
save/RestoreV2_10/tensor_namesConst*
dtype0*5
value,B*B linear/linear_model/beta/weights*
_output_shapes
:
v
"save/RestoreV2_10/shape_and_slicesConst*
dtype0* 
valueBB2 1 0,2:0,1*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
т
save/Assign_10Assign'linear/linear_model/beta/weights/part_0save/RestoreV2_10*
validate_shape(*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
use_locking(*
T0*
_output_shapes

:
З
save/RestoreV2_11/tensor_namesConst*
dtype0*5
value,B*B linear/linear_model/bias_weights*
_output_shapes
:
p
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
▐
save/Assign_11Assign'linear/linear_model/bias_weights/part_0save/RestoreV2_11*
validate_shape(*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
use_locking(*
T0*
_output_shapes
:
┌
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11
-
save/restore_allNoOp^save/restore_shard"(!O U     g ╜х	Й╧╩< l╓AJЄл
г<Б<
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	АР
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
╢
AsString

input"T

output"
Ttype:
	2	
"
	precisionint         "

scientificbool( "
shortestbool( "
widthint         "
fillstring 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignAdd
ref"TА

value"T

output_ref"TА"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Л
	DecodeCSV
records
record_defaults2OUT_TYPE
output2OUT_TYPE"$
OUT_TYPE
list(type)(0:
2	"
field_delimstring,
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
+
Exp
x"T
y"T"
Ttype:	
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
о
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint         "
	containerstring "
shared_namestring И
4
Fill
dims

value"T
output"T"	
Ttype
М
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
в
	HashTable
table_handleА"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
б
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
`
InitializeTable
table_handleА
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
b
InitializeTableV2
table_handle
keys"Tkey
values"Tval"
Tkeytype"
TvaltypeИ
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
-
Log1p
x"T
y"T"
Ttype:	
2
$

LogicalAnd
x

y

z
Р


LogicalNot
x

y

u
LookupTableFind
table_handleА
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint         "	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
B
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( И
М
QueueDequeueUpToV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint         И
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint         И
&
QueueSizeV2

handle
sizeИ
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
a
ReaderReadUpToV2
reader_handle
queue_handle
num_records	
keys

valuesИ
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sigmoid
x"T
y"T"
Ttype:	
2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
y
SparseReorder
input_indices	
input_values"T
input_shape	
output_indices	
output_values"T"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
А
SparseSegmentSum	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2		"
Tidxtype0:
2	
╝
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
z
TextLineReaderV2
reader_handle"
skip_header_linesint "
	containerstring "
shared_namestring И
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.2.02v1.2.0-rc2-21-g12f033dТ╡

global_step/Initializer/zerosConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
П
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
s
input_producer/ConstConst*
dtype0*+
value"B B../data/valid-data.csv*
_output_shapes
:
U
input_producer/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Z
input_producer/Greater/yConst*
dtype0*
value	B : *
_output_shapes
: 
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 
Т
input_producer/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
Ъ
#input_producer/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor*
_output_shapes
: 
А
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*
	summarize*

T
2
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
У
input_producerFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
Щ
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/Identity*

timeout_ms         *
Tcomponents
2
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
Y
input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 
К
'input_producer/fraction_of_32_full/tagsConst*
dtype0*3
value*B( B"input_producer/fraction_of_32_full*
_output_shapes
: 
С
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
y
TextLineReaderV2TextLineReaderV2*
shared_name *
	container *
skip_header_lines *
_output_shapes
: 
_
ReaderReadUpToV2/num_recordsConst*
dtype0	*
value
B	 RЇ*
_output_shapes
: 
Ш
ReaderReadUpToV2ReaderReadUpToV2TextLineReaderV2input_producerReaderReadUpToV2/num_records*2
_output_shapes 
:         :         
Y
ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
z

ExpandDims
ExpandDimsReaderReadUpToV2:1ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
h
DecodeCSV/record_defaults_0Const*
dtype0*
valueB*    *
_output_shapes
:
h
DecodeCSV/record_defaults_1Const*
dtype0*
valueB*    *
_output_shapes
:
h
DecodeCSV/record_defaults_2Const*
dtype0*
valueB*    *
_output_shapes
:
d
DecodeCSV/record_defaults_3Const*
dtype0*
valueB
B *
_output_shapes
:
d
DecodeCSV/record_defaults_4Const*
dtype0*
valueB
B *
_output_shapes
:
d
DecodeCSV/record_defaults_5Const*
dtype0*
valueB
B *
_output_shapes
:
Е
	DecodeCSV	DecodeCSV
ExpandDimsDecodeCSV/record_defaults_0DecodeCSV/record_defaults_1DecodeCSV/record_defaults_2DecodeCSV/record_defaults_3DecodeCSV/record_defaults_4DecodeCSV/record_defaults_5*
OUT_TYPE

2*
field_delim,*Ж
_output_shapest
r:         :         :         :         :         :         
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
╢
batch/fifo_queueFIFOQueueV2*
capacityИ'*
component_types	
2*
_output_shapes
: **
shapes 
:::::*
	container *
shared_name 
║
batch/fifo_queue_EnqueueManyQueueEnqueueManyV2batch/fifo_queueDecodeCSV:3DecodeCSV:4DecodeCSV:5DecodeCSV:1DecodeCSV:2*

timeout_ms         *
Tcomponents	
2
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *╖Q9*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
|
 batch/fraction_of_5000_full/tagsConst*
dtype0*,
value#B! Bbatch/fraction_of_5000_full*
_output_shapes
: 
z
batch/fraction_of_5000_fullScalarSummary batch/fraction_of_5000_full/tags	batch/mul*
T0*
_output_shapes
: 
J
batch/nConst*
dtype0*
value
B :Ї*
_output_shapes
: 
ф
batchQueueDequeueUpToV2batch/fifo_queuebatch/n*

timeout_ms         *
component_types	
2*s
_output_shapesa
_:         :         :         :         :         
`
ConstConst*
dtype0*'
valueBBpositiveBnegative*
_output_shapes
:
V
string_to_index/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
]
string_to_index/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
]
string_to_index/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ц
string_to_index/rangeRangestring_to_index/range/startstring_to_index/Sizestring_to_index/range/delta*

Tidx0*
_output_shapes
:
j
string_to_index/ToInt64Caststring_to_index/range*

DstT0	*

SrcT0*
_output_shapes
:
и
string_to_index/hash_table	HashTable*
	container *
	key_dtype0*
_output_shapes
:*
use_node_name_sharing( *
value_dtype0	*
shared_name 
k
 string_to_index/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
╗
%string_to_index/hash_table/table_initInitializeTablestring_to_index/hash_tableConststring_to_index/ToInt64*-
_class#
!loc:@string_to_index/hash_table*

Tkey0*

Tval0	
┌
hash_table_LookupLookupTableFindstring_to_index/hash_tablebatch:2 string_to_index/hash_table/Const*	
Tin0*-
_class#
!loc:@string_to_index/hash_table*

Tout0	*'
_output_shapes
:         
Х
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ShapeShapebatch*
out_type0*
T0*
_output_shapes
:
▌
Odnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/CastCastPdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
Ф
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
э
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/NotEqualNotEqualbatchSdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╫
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/WhereWhereSdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/NotEqual*'
_output_shapes
:         
л
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
·
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ReshapeReshapebatchXdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Reshape/shape*
Tshape0*
T0*#
_output_shapes
:         
п
^dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
¤
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_sliceStridedSlicePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Where^dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_1`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
▒
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
│
bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
│
bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Й
Zdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1StridedSlicePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Where`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stackbdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_1bdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
ч
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/unstackUnpackOdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Cast*	
num*

axis *
T0	*
_output_shapes
: : 
ш
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/stackPackTdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/unstack:1*
_output_shapes
:*

axis *
T0	*
N
╡
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/MulMulZdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_slice_1Pdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/stack*
T0	*'
_output_shapes
:         
к
`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
╥
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/SumSumNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Mul`dnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum/reduction_indices*

Tidx0*
T0	*
	keep_dims( *#
_output_shapes
:         
н
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/AddAddXdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/strided_sliceNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
█
Qdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/GatherGatherRdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/ReshapeNdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
а
Mdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ConstConst*
dtype0*
valueBBax01Bax02*
_output_shapes
:
О
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Х
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Х
Sdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ў
Mdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/rangeRangeSdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/startLdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/SizeSdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range/delta*

Tidx0*
_output_shapes
:
┌
Odnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ToInt64CastMdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
▐
Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
г
Xdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
■
]dnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/table_initInitializeTableV2Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableMdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ConstOdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/ToInt64*

Tkey0*

Tval0	
Ю
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/hash_table_LookupLookupTableFindV2Rdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_tableQdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/GatherXdnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
б
Vdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDense/default_valueConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Е
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDenseSparseToDensePdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/WhereOdnn/input_from_feature_columns/input_layer/alpha_indicator/to_sparse_input/CastLdnn/input_from_feature_columns/input_layer/alpha_indicator/hash_table_LookupVdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0	*0
_output_shapes
:                  
Н
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
П
Jdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
К
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
Р
Kdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
С
Ldnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┘
Bdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hotOneHotHdnn/input_from_feature_columns/input_layer/alpha_indicator/SparseToDenseHdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/depthKdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/on_valueLdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hot/off_value*
TI0	*4
_output_shapes"
 :                  *
T0*
axis         
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
к
>dnn/input_from_feature_columns/input_layer/alpha_indicator/SumSumBdnn/input_from_feature_columns/input_layer/alpha_indicator/one_hotPdnn/input_from_feature_columns/input_layer/alpha_indicator/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *'
_output_shapes
:         
╛
@dnn/input_from_feature_columns/input_layer/alpha_indicator/ShapeShape>dnn/input_from_feature_columns/input_layer/alpha_indicator/Sum*
out_type0*
T0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
а
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/alpha_indicator/ShapeNdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stackPdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
М
Jdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
а
Hdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/alpha_indicator/strided_sliceJdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
Ч
Bdnn/input_from_feature_columns/input_layer/alpha_indicator/ReshapeReshape>dnn/input_from_feature_columns/input_layer/alpha_indicator/SumHdnn/input_from_feature_columns/input_layer/alpha_indicator/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Ц
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ShapeShapebatch:1*
out_type0*
T0*
_output_shapes
:
█
Ndnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/CastCastOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
У
Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
э
Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/NotEqualNotEqualbatch:1Rdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╒
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/WhereWhereRdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/NotEqual*'
_output_shapes
:         
к
Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
·
Qdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ReshapeReshapebatch:1Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Reshape/shape*
Tshape0*
T0*#
_output_shapes
:         
о
]dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
°
Wdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_sliceStridedSliceOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Where]dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_1_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
░
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Д
Ydnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1StridedSliceOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Where_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stackadnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_1adnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
х
Qdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/unstackUnpackNdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Cast*	
num*

axis *
T0	*
_output_shapes
: : 
ц
Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/stackPackSdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/unstack:1*
_output_shapes
:*

axis *
T0	*
N
▓
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/MulMulYdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_slice_1Odnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/stack*
T0	*'
_output_shapes
:         
й
_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
╧
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/SumSumMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Mul_dnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum/reduction_indices*

Tidx0*
T0	*
	keep_dims( *#
_output_shapes
:         
к
Mdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/AddAddWdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/strided_sliceMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
╪
Pdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/GatherGatherQdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/ReshapeMdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
Ю
Kdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ConstConst*
dtype0*
valueBBbx01Bbx02*
_output_shapes
:
М
Jdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
У
Qdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
У
Qdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ю
Kdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/rangeRangeQdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/startJdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/SizeQdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range/delta*

Tidx0*
_output_shapes
:
╓
Mdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ToInt64CastKdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
▄
Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
б
Vdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Ў
[dnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/table_initInitializeTableV2Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tableKdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ConstMdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/ToInt64*

Tkey0*

Tval0	
Ш
Kdnn/input_from_feature_columns/input_layer/beta_indicator/hash_table_LookupLookupTableFindV2Pdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_tablePdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/GatherVdnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
а
Udnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDense/default_valueConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
А
Gdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDenseSparseToDenseOdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/WhereNdnn/input_from_feature_columns/input_layer/beta_indicator/to_sparse_input/CastKdnn/input_from_feature_columns/input_layer/beta_indicator/hash_table_LookupUdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0	*0
_output_shapes
:                  
М
Gdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
О
Idnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Й
Gdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
П
Jdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/on_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
Р
Kdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╘
Adnn/input_from_feature_columns/input_layer/beta_indicator/one_hotOneHotGdnn/input_from_feature_columns/input_layer/beta_indicator/SparseToDenseGdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/depthJdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/on_valueKdnn/input_from_feature_columns/input_layer/beta_indicator/one_hot/off_value*
TI0	*4
_output_shapes"
 :                  *
T0*
axis         
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
з
=dnn/input_from_feature_columns/input_layer/beta_indicator/SumSumAdnn/input_from_feature_columns/input_layer/beta_indicator/one_hotOdnn/input_from_feature_columns/input_layer/beta_indicator/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *'
_output_shapes
:         
╝
?dnn/input_from_feature_columns/input_layer/beta_indicator/ShapeShape=dnn/input_from_feature_columns/input_layer/beta_indicator/Sum*
out_type0*
T0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/beta_indicator/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/beta_indicator/ShapeMdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stackOdnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/beta_indicator/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Л
Idnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/beta_indicator/strided_sliceIdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
Ф
Adnn/input_from_feature_columns/input_layer/beta_indicator/ReshapeReshape=dnn/input_from_feature_columns/input_layer/beta_indicator/SumGdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
y
2dnn/input_from_feature_columns/input_layer/x/ShapeShapebatch:3*
out_type0*
T0*
_output_shapes
:
К
@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┌
:dnn/input_from_feature_columns/input_layer/x/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/x/Shape@dnn/input_from_feature_columns/input_layer/x/strided_slice/stackBdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/x/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
~
<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ў
:dnn/input_from_feature_columns/input_layer/x/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/x/strided_slice<dnn/input_from_feature_columns/input_layer/x/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
─
4dnn/input_from_feature_columns/input_layer/x/ReshapeReshapebatch:3:dnn/input_from_feature_columns/input_layer/x/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
y
2dnn/input_from_feature_columns/input_layer/y/ShapeShapebatch:4*
out_type0*
T0*
_output_shapes
:
К
@dnn/input_from_feature_columns/input_layer/y/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
М
Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┌
:dnn/input_from_feature_columns/input_layer/y/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/y/Shape@dnn/input_from_feature_columns/input_layer/y/strided_slice/stackBdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/y/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
~
<dnn/input_from_feature_columns/input_layer/y/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ў
:dnn/input_from_feature_columns/input_layer/y/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/y/strided_slice<dnn/input_from_feature_columns/input_layer/y/Reshape/shape/1*
_output_shapes
:*

axis *
T0*
N
─
4dnn/input_from_feature_columns/input_layer/y/ReshapeReshapebatch:4:dnn/input_from_feature_columns/input_layer/y/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
п
1dnn/input_from_feature_columns/input_layer/concatConcatV2Bdnn/input_from_feature_columns/input_layer/alpha_indicator/ReshapeAdnn/input_from_feature_columns/input_layer/beta_indicator/Reshape4dnn/input_from_feature_columns/input_layer/x/Reshape4dnn/input_from_feature_columns/input_layer/y/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*'
_output_shapes
:         *

Tidx0*
T0*
N
╟
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"   А   *
_output_shapes
:
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *гоX╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *гоX>*
_output_shapes
: 
в
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes
:	А*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
Ю
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
▒
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
г
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
╦
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes
:	А*
dtype0*
shape:	А*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
Ш
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А
▓
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
:	А
┤
1dnn/hiddenlayer_0/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBА*    *
_output_shapes	
:А
┴
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes	
:А*
dtype0*
shape:А*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
З
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes	
:А
л
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes	
:А
v
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes
:	А
╔
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
p
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes	
:А
в
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
data_formatNHWC*
T0*(
_output_shapes
:         А
z
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:         А
[
dnn/zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
С
dnn/zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*(
_output_shapes
:         А
y
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

DstT0*

SrcT0
*(
_output_shapes
:         А
h
dnn/zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Н
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
_output_shapes
: 
л
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"А   @   *
_output_shapes
:
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *є5╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *є5>*
_output_shapes
: 
в
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes
:	А@*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
Ю
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
▒
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
г
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
╦
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes
:	А@*
dtype0*
shape:	А@*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
Ш
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А@
▓
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
:	А@
▓
1dnn/hiddenlayer_1/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueB@*    *
_output_shapes
:@
┐
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:@
к
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:@
v
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes
:	А@
╗
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         @
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:@
б
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
data_formatNHWC*
T0*'
_output_shapes
:         @
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         @
]
dnn/zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ф
dnn/zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:         @
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         @
j
dnn/zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
╟
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"@       *
_output_shapes
:
╣
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *  А╛*
_output_shapes
: 
╣
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *  А>*
_output_shapes
: 
б
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:@ *
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
Ю
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
░
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
в
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
╔
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:@ *
dtype0*
shape
:@ *3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
Ч
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:@ 
▒
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:@ 
▓
1dnn/hiddenlayer_2/biases/part_0/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB *    *
_output_shapes
: 
┐
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
Ж
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/zeros*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
: 
к
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
: 
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:@ 
╗
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:          
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
: 
б
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*'
_output_shapes
:          
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:          
]
dnn/zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Ф
dnn/zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:          
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:          
j
dnn/zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_2/activation/tagConst*
dtype0*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
_output_shapes
: 
б
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
╣
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"       *
_output_shapes
:
л
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *JQ┌╛*
_output_shapes
: 
л
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *JQ┌>*
_output_shapes
: 
М
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

: *
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
В
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
Ф
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
Ж
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
╗
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

: *
dtype0*
shape
: *,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
√
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

: 
Ь
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

: 
д
*dnn/logits/biases/part_0/Initializer/zerosConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
▒
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
ъ
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/zeros*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
Х
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

: 
н
dnn/logits/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
М
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*'
_output_shapes
:         
]
dnn/zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
В
dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:         
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
j
dnn/zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
У
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
dtype0*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
_output_shapes
: 
Я
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst*
dtype0**
value!B Bdnn/dnn/logits/activation*
_output_shapes
: 
Б
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
t
/linear/linear_model/alpha/to_sparse_input/ShapeShapebatch*
out_type0*
T0*
_output_shapes
:
Ы
.linear/linear_model/alpha/to_sparse_input/CastCast/linear/linear_model/alpha/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
s
2linear/linear_model/alpha/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
л
2linear/linear_model/alpha/to_sparse_input/NotEqualNotEqualbatch2linear/linear_model/alpha/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
Х
/linear/linear_model/alpha/to_sparse_input/WhereWhere2linear/linear_model/alpha/to_sparse_input/NotEqual*'
_output_shapes
:         
К
7linear/linear_model/alpha/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╕
1linear/linear_model/alpha/to_sparse_input/ReshapeReshapebatch7linear/linear_model/alpha/to_sparse_input/Reshape/shape*
Tshape0*
T0*#
_output_shapes
:         
О
=linear/linear_model/alpha/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╪
7linear/linear_model/alpha/to_sparse_input/strided_sliceStridedSlice/linear/linear_model/alpha/to_sparse_input/Where=linear/linear_model/alpha/to_sparse_input/strided_slice/stack?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_1?linear/linear_model/alpha/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
Р
?linear/linear_model/alpha/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
Т
Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
Т
Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ф
9linear/linear_model/alpha/to_sparse_input/strided_slice_1StridedSlice/linear/linear_model/alpha/to_sparse_input/Where?linear/linear_model/alpha/to_sparse_input/strided_slice_1/stackAlinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_1Alinear/linear_model/alpha/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
е
1linear/linear_model/alpha/to_sparse_input/unstackUnpack.linear/linear_model/alpha/to_sparse_input/Cast*	
num*

axis *
T0	*
_output_shapes
: : 
ж
/linear/linear_model/alpha/to_sparse_input/stackPack3linear/linear_model/alpha/to_sparse_input/unstack:1*
_output_shapes
:*

axis *
T0	*
N
╥
-linear/linear_model/alpha/to_sparse_input/MulMul9linear/linear_model/alpha/to_sparse_input/strided_slice_1/linear/linear_model/alpha/to_sparse_input/stack*
T0	*'
_output_shapes
:         
Й
?linear/linear_model/alpha/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
я
-linear/linear_model/alpha/to_sparse_input/SumSum-linear/linear_model/alpha/to_sparse_input/Mul?linear/linear_model/alpha/to_sparse_input/Sum/reduction_indices*

Tidx0*
T0	*
	keep_dims( *#
_output_shapes
:         
╩
-linear/linear_model/alpha/to_sparse_input/AddAdd7linear/linear_model/alpha/to_sparse_input/strided_slice-linear/linear_model/alpha/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
°
0linear/linear_model/alpha/to_sparse_input/GatherGather1linear/linear_model/alpha/to_sparse_input/Reshape-linear/linear_model/alpha/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         

,linear/linear_model/alpha/alpha_lookup/ConstConst*
dtype0*
valueBBax01Bax02*
_output_shapes
:
m
+linear/linear_model/alpha/alpha_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
t
2linear/linear_model/alpha/alpha_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
t
2linear/linear_model/alpha/alpha_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Є
,linear/linear_model/alpha/alpha_lookup/rangeRange2linear/linear_model/alpha/alpha_lookup/range/start+linear/linear_model/alpha/alpha_lookup/Size2linear/linear_model/alpha/alpha_lookup/range/delta*

Tidx0*
_output_shapes
:
Ш
.linear/linear_model/alpha/alpha_lookup/ToInt64Cast,linear/linear_model/alpha/alpha_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
╜
1linear/linear_model/alpha/alpha_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
В
7linear/linear_model/alpha/alpha_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
·
<linear/linear_model/alpha/alpha_lookup/hash_table/table_initInitializeTableV21linear/linear_model/alpha/alpha_lookup/hash_table,linear/linear_model/alpha/alpha_lookup/Const.linear/linear_model/alpha/alpha_lookup/ToInt64*

Tkey0*

Tval0	
Ъ
+linear/linear_model/alpha/hash_table_LookupLookupTableFindV21linear/linear_model/alpha/alpha_lookup/hash_table0linear/linear_model/alpha/to_sparse_input/Gather7linear/linear_model/alpha/alpha_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
Р
$linear/linear_model/alpha/Shape/CastCast.linear/linear_model/alpha/to_sparse_input/Cast*

DstT0*

SrcT0	*
_output_shapes
:
w
-linear/linear_model/alpha/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
y
/linear/linear_model/alpha/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
y
/linear/linear_model/alpha/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
'linear/linear_model/alpha/strided_sliceStridedSlice$linear/linear_model/alpha/Shape/Cast-linear/linear_model/alpha/strided_slice/stack/linear/linear_model/alpha/strided_slice/stack_1/linear/linear_model/alpha/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
m
"linear/linear_model/alpha/Cast/x/1Const*
dtype0*
valueB :
         *
_output_shapes
: 
п
 linear/linear_model/alpha/Cast/xPack'linear/linear_model/alpha/strided_slice"linear/linear_model/alpha/Cast/x/1*
_output_shapes
:*

axis *
T0*
N
|
linear/linear_model/alpha/CastCast linear/linear_model/alpha/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
ш
'linear/linear_model/alpha/SparseReshapeSparseReshape/linear/linear_model/alpha/to_sparse_input/Where.linear/linear_model/alpha/to_sparse_input/Castlinear/linear_model/alpha/Cast*-
_output_shapes
:         :
Ч
0linear/linear_model/alpha/SparseReshape/IdentityIdentity+linear/linear_model/alpha/hash_table_Lookup*
T0	*#
_output_shapes
:         
╠
:linear/linear_model/alpha/weights/part_0/Initializer/zerosConst*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
valueB*    *
_output_shapes

:
┘
(linear/linear_model/alpha/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
shared_name 
о
/linear/linear_model/alpha/weights/part_0/AssignAssign(linear/linear_model/alpha/weights/part_0:linear/linear_model/alpha/weights/part_0/Initializer/zeros*
validate_shape(*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
use_locking(*
T0*
_output_shapes

:
╔
-linear/linear_model/alpha/weights/part_0/readIdentity(linear/linear_model/alpha/weights/part_0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
T0*
_output_shapes

:
|
2linear/linear_model/alpha/weighted_sum/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
{
1linear/linear_model/alpha/weighted_sum/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
∙
,linear/linear_model/alpha/weighted_sum/SliceSlice)linear/linear_model/alpha/SparseReshape:12linear/linear_model/alpha/weighted_sum/Slice/begin1linear/linear_model/alpha/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
v
,linear/linear_model/alpha/weighted_sum/ConstConst*
dtype0*
valueB: *
_output_shapes
:
═
+linear/linear_model/alpha/weighted_sum/ProdProd,linear/linear_model/alpha/weighted_sum/Slice,linear/linear_model/alpha/weighted_sum/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
w
5linear/linear_model/alpha/weighted_sum/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
ш
-linear/linear_model/alpha/weighted_sum/GatherGather)linear/linear_model/alpha/SparseReshape:15linear/linear_model/alpha/weighted_sum/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
╦
-linear/linear_model/alpha/weighted_sum/Cast/xPack+linear/linear_model/alpha/weighted_sum/Prod-linear/linear_model/alpha/weighted_sum/Gather*
_output_shapes
:*

axis *
T0	*
N
ў
4linear/linear_model/alpha/weighted_sum/SparseReshapeSparseReshape'linear/linear_model/alpha/SparseReshape)linear/linear_model/alpha/SparseReshape:1-linear/linear_model/alpha/weighted_sum/Cast/x*-
_output_shapes
:         :
й
=linear/linear_model/alpha/weighted_sum/SparseReshape/IdentityIdentity0linear/linear_model/alpha/SparseReshape/Identity*
T0	*#
_output_shapes
:         
w
5linear/linear_model/alpha/weighted_sum/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ч
3linear/linear_model/alpha/weighted_sum/GreaterEqualGreaterEqual=linear/linear_model/alpha/weighted_sum/SparseReshape/Identity5linear/linear_model/alpha/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:         
У
,linear/linear_model/alpha/weighted_sum/WhereWhere3linear/linear_model/alpha/weighted_sum/GreaterEqual*'
_output_shapes
:         
З
4linear/linear_model/alpha/weighted_sum/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
┘
.linear/linear_model/alpha/weighted_sum/ReshapeReshape,linear/linear_model/alpha/weighted_sum/Where4linear/linear_model/alpha/weighted_sum/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
 
/linear/linear_model/alpha/weighted_sum/Gather_1Gather4linear/linear_model/alpha/weighted_sum/SparseReshape.linear/linear_model/alpha/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Д
/linear/linear_model/alpha/weighted_sum/Gather_2Gather=linear/linear_model/alpha/weighted_sum/SparseReshape/Identity.linear/linear_model/alpha/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
Ш
/linear/linear_model/alpha/weighted_sum/IdentityIdentity6linear/linear_model/alpha/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
В
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ш
Nlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
П
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_sliceStridedSlice/linear/linear_model/alpha/weighted_sum/IdentityNlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stackPlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
┴
?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/CastCastHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╦
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/rangeRangeFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/start?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/CastFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╚
Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Cast_1Cast@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
б
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
д
Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1StridedSlice/linear/linear_model/alpha/weighted_sum/Gather_1Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackRlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
к
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffListDiffAlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Cast_1Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
Ъ
Plinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ь
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ь
Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ч
Jlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2StridedSlice/linear/linear_model/alpha/weighted_sum/IdentityPlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackRlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Rlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
Ф
Ilinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Ы
Elinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims
ExpandDimsJlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/strided_slice_2Ilinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Ш
Vlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ш
Vlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ы
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDenseSparseToDenseClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffElinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ExpandDimsVlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesVlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
Щ
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ь
Blinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ReshapeReshapeClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiffHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:         
╚
Elinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/zeros_like	ZerosLikeBlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
И
Flinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
ч
Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concatConcatV2Blinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ReshapeElinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/zeros_likeFlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
├
@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ShapeShapeClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
∙
?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/FillFill@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Shape@linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
К
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╘
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1ConcatV2/linear/linear_model/alpha/weighted_sum/Gather_1Alinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concatHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
К
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╬
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2ConcatV2/linear/linear_model/alpha/weighted_sum/Gather_2?linear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/FillHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:         *

Tidx0*
T0	*
N
╒
Hlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorderSparseReorderClinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_1Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/concat_2/linear/linear_model/alpha/weighted_sum/Identity*
T0	*6
_output_shapes$
":         :         
е
Clinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/IdentityIdentity/linear/linear_model/alpha/weighted_sum/Identity*
T0	*
_output_shapes
:
г
Rlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
е
Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
е
Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
┼
Llinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorderRlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stackTlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Tlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
╓
Clinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/CastCastLlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
ч
Elinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/UniqueUniqueJlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:         :         
ь
Olinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/embedding_lookupGather-linear/linear_model/alpha/weights/part_0/readElinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Unique*
validate_indices(*
Tparams0*
Tindices0	*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*'
_output_shapes
:         
я
>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparseSparseSegmentSumOlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/embedding_lookupGlinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Unique:1Clinear/linear_model/alpha/weighted_sum/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:         
З
6linear/linear_model/alpha/weighted_sum/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
¤
0linear/linear_model/alpha/weighted_sum/Reshape_1ReshapeHlinear/linear_model/alpha/weighted_sum/SparseFillEmptyRows/SparseToDense6linear/linear_model/alpha/weighted_sum/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
к
,linear/linear_model/alpha/weighted_sum/ShapeShape>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Д
:linear/linear_model/alpha/weighted_sum/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Ж
<linear/linear_model/alpha/weighted_sum/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ж
<linear/linear_model/alpha/weighted_sum/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╝
4linear/linear_model/alpha/weighted_sum/strided_sliceStridedSlice,linear/linear_model/alpha/weighted_sum/Shape:linear/linear_model/alpha/weighted_sum/strided_slice/stack<linear/linear_model/alpha/weighted_sum/strided_slice/stack_1<linear/linear_model/alpha/weighted_sum/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
p
.linear/linear_model/alpha/weighted_sum/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╘
,linear/linear_model/alpha/weighted_sum/stackPack.linear/linear_model/alpha/weighted_sum/stack/04linear/linear_model/alpha/weighted_sum/strided_slice*
_output_shapes
:*

axis *
T0*
N
р
+linear/linear_model/alpha/weighted_sum/TileTile0linear/linear_model/alpha/weighted_sum/Reshape_1,linear/linear_model/alpha/weighted_sum/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
░
1linear/linear_model/alpha/weighted_sum/zeros_like	ZerosLike>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
В
&linear/linear_model/alpha/weighted_sumSelect+linear/linear_model/alpha/weighted_sum/Tile1linear/linear_model/alpha/weighted_sum/zeros_like>linear/linear_model/alpha/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Ф
-linear/linear_model/alpha/weighted_sum/Cast_1Cast)linear/linear_model/alpha/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
~
4linear/linear_model/alpha/weighted_sum/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
}
3linear/linear_model/alpha/weighted_sum/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
Г
.linear/linear_model/alpha/weighted_sum/Slice_1Slice-linear/linear_model/alpha/weighted_sum/Cast_14linear/linear_model/alpha/weighted_sum/Slice_1/begin3linear/linear_model/alpha/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ф
.linear/linear_model/alpha/weighted_sum/Shape_1Shape&linear/linear_model/alpha/weighted_sum*
out_type0*
T0*
_output_shapes
:
~
4linear/linear_model/alpha/weighted_sum/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Ж
3linear/linear_model/alpha/weighted_sum/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
Д
.linear/linear_model/alpha/weighted_sum/Slice_2Slice.linear/linear_model/alpha/weighted_sum/Shape_14linear/linear_model/alpha/weighted_sum/Slice_2/begin3linear/linear_model/alpha/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
t
2linear/linear_model/alpha/weighted_sum/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
З
-linear/linear_model/alpha/weighted_sum/concatConcatV2.linear/linear_model/alpha/weighted_sum/Slice_1.linear/linear_model/alpha/weighted_sum/Slice_22linear/linear_model/alpha/weighted_sum/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╥
0linear/linear_model/alpha/weighted_sum/Reshape_2Reshape&linear/linear_model/alpha/weighted_sum-linear/linear_model/alpha/weighted_sum/concat*
Tshape0*
T0*'
_output_shapes
:         
u
.linear/linear_model/beta/to_sparse_input/ShapeShapebatch:1*
out_type0*
T0*
_output_shapes
:
Щ
-linear/linear_model/beta/to_sparse_input/CastCast.linear/linear_model/beta/to_sparse_input/Shape*

DstT0	*

SrcT0*
_output_shapes
:
r
1linear/linear_model/beta/to_sparse_input/Cast_1/xConst*
dtype0*
valueB B *
_output_shapes
: 
л
1linear/linear_model/beta/to_sparse_input/NotEqualNotEqualbatch:11linear/linear_model/beta/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
У
.linear/linear_model/beta/to_sparse_input/WhereWhere1linear/linear_model/beta/to_sparse_input/NotEqual*'
_output_shapes
:         
Й
6linear/linear_model/beta/to_sparse_input/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╕
0linear/linear_model/beta/to_sparse_input/ReshapeReshapebatch:16linear/linear_model/beta/to_sparse_input/Reshape/shape*
Tshape0*
T0*#
_output_shapes
:         
Н
<linear/linear_model/beta/to_sparse_input/strided_slice/stackConst*
dtype0*
valueB"       *
_output_shapes
:
П
>linear/linear_model/beta/to_sparse_input/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
П
>linear/linear_model/beta/to_sparse_input/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
╙
6linear/linear_model/beta/to_sparse_input/strided_sliceStridedSlice.linear/linear_model/beta/to_sparse_input/Where<linear/linear_model/beta/to_sparse_input/strided_slice/stack>linear/linear_model/beta/to_sparse_input/strided_slice/stack_1>linear/linear_model/beta/to_sparse_input/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
П
>linear/linear_model/beta/to_sparse_input/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
С
@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
С
@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
▀
8linear/linear_model/beta/to_sparse_input/strided_slice_1StridedSlice.linear/linear_model/beta/to_sparse_input/Where>linear/linear_model/beta/to_sparse_input/strided_slice_1/stack@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_1@linear/linear_model/beta/to_sparse_input/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask 
г
0linear/linear_model/beta/to_sparse_input/unstackUnpack-linear/linear_model/beta/to_sparse_input/Cast*	
num*

axis *
T0	*
_output_shapes
: : 
д
.linear/linear_model/beta/to_sparse_input/stackPack2linear/linear_model/beta/to_sparse_input/unstack:1*
_output_shapes
:*

axis *
T0	*
N
╧
,linear/linear_model/beta/to_sparse_input/MulMul8linear/linear_model/beta/to_sparse_input/strided_slice_1.linear/linear_model/beta/to_sparse_input/stack*
T0	*'
_output_shapes
:         
И
>linear/linear_model/beta/to_sparse_input/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
ь
,linear/linear_model/beta/to_sparse_input/SumSum,linear/linear_model/beta/to_sparse_input/Mul>linear/linear_model/beta/to_sparse_input/Sum/reduction_indices*

Tidx0*
T0	*
	keep_dims( *#
_output_shapes
:         
╟
,linear/linear_model/beta/to_sparse_input/AddAdd6linear/linear_model/beta/to_sparse_input/strided_slice,linear/linear_model/beta/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
ї
/linear/linear_model/beta/to_sparse_input/GatherGather0linear/linear_model/beta/to_sparse_input/Reshape,linear/linear_model/beta/to_sparse_input/Add*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
}
*linear/linear_model/beta/beta_lookup/ConstConst*
dtype0*
valueBBbx01Bbx02*
_output_shapes
:
k
)linear/linear_model/beta/beta_lookup/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
r
0linear/linear_model/beta/beta_lookup/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
r
0linear/linear_model/beta/beta_lookup/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
ъ
*linear/linear_model/beta/beta_lookup/rangeRange0linear/linear_model/beta/beta_lookup/range/start)linear/linear_model/beta/beta_lookup/Size0linear/linear_model/beta/beta_lookup/range/delta*

Tidx0*
_output_shapes
:
Ф
,linear/linear_model/beta/beta_lookup/ToInt64Cast*linear/linear_model/beta/beta_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
╗
/linear/linear_model/beta/beta_lookup/hash_tableHashTableV2*
	container *
	key_dtype0*
_output_shapes
: *
use_node_name_sharing( *
value_dtype0	*
shared_name 
А
5linear/linear_model/beta/beta_lookup/hash_table/ConstConst*
dtype0	*
valueB	 R
         *
_output_shapes
: 
Є
:linear/linear_model/beta/beta_lookup/hash_table/table_initInitializeTableV2/linear/linear_model/beta/beta_lookup/hash_table*linear/linear_model/beta/beta_lookup/Const,linear/linear_model/beta/beta_lookup/ToInt64*

Tkey0*

Tval0	
Ф
*linear/linear_model/beta/hash_table_LookupLookupTableFindV2/linear/linear_model/beta/beta_lookup/hash_table/linear/linear_model/beta/to_sparse_input/Gather5linear/linear_model/beta/beta_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:         
О
#linear/linear_model/beta/Shape/CastCast-linear/linear_model/beta/to_sparse_input/Cast*

DstT0*

SrcT0	*
_output_shapes
:
v
,linear/linear_model/beta/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
x
.linear/linear_model/beta/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
x
.linear/linear_model/beta/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
√
&linear/linear_model/beta/strided_sliceStridedSlice#linear/linear_model/beta/Shape/Cast,linear/linear_model/beta/strided_slice/stack.linear/linear_model/beta/strided_slice/stack_1.linear/linear_model/beta/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
l
!linear/linear_model/beta/Cast/x/1Const*
dtype0*
valueB :
         *
_output_shapes
: 
м
linear/linear_model/beta/Cast/xPack&linear/linear_model/beta/strided_slice!linear/linear_model/beta/Cast/x/1*
_output_shapes
:*

axis *
T0*
N
z
linear/linear_model/beta/CastCastlinear/linear_model/beta/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
ф
&linear/linear_model/beta/SparseReshapeSparseReshape.linear/linear_model/beta/to_sparse_input/Where-linear/linear_model/beta/to_sparse_input/Castlinear/linear_model/beta/Cast*-
_output_shapes
:         :
Х
/linear/linear_model/beta/SparseReshape/IdentityIdentity*linear/linear_model/beta/hash_table_Lookup*
T0	*#
_output_shapes
:         
╩
9linear/linear_model/beta/weights/part_0/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
valueB*    *
_output_shapes

:
╫
'linear/linear_model/beta/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
shared_name 
к
.linear/linear_model/beta/weights/part_0/AssignAssign'linear/linear_model/beta/weights/part_09linear/linear_model/beta/weights/part_0/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
use_locking(*
T0*
_output_shapes

:
╞
,linear/linear_model/beta/weights/part_0/readIdentity'linear/linear_model/beta/weights/part_0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
T0*
_output_shapes

:
{
1linear/linear_model/beta/weighted_sum/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
z
0linear/linear_model/beta/weighted_sum/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ї
+linear/linear_model/beta/weighted_sum/SliceSlice(linear/linear_model/beta/SparseReshape:11linear/linear_model/beta/weighted_sum/Slice/begin0linear/linear_model/beta/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
u
+linear/linear_model/beta/weighted_sum/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╩
*linear/linear_model/beta/weighted_sum/ProdProd+linear/linear_model/beta/weighted_sum/Slice+linear/linear_model/beta/weighted_sum/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
v
4linear/linear_model/beta/weighted_sum/Gather/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
х
,linear/linear_model/beta/weighted_sum/GatherGather(linear/linear_model/beta/SparseReshape:14linear/linear_model/beta/weighted_sum/Gather/indices*
validate_indices(*
Tparams0	*
Tindices0*
_output_shapes
: 
╚
,linear/linear_model/beta/weighted_sum/Cast/xPack*linear/linear_model/beta/weighted_sum/Prod,linear/linear_model/beta/weighted_sum/Gather*
_output_shapes
:*

axis *
T0	*
N
є
3linear/linear_model/beta/weighted_sum/SparseReshapeSparseReshape&linear/linear_model/beta/SparseReshape(linear/linear_model/beta/SparseReshape:1,linear/linear_model/beta/weighted_sum/Cast/x*-
_output_shapes
:         :
з
<linear/linear_model/beta/weighted_sum/SparseReshape/IdentityIdentity/linear/linear_model/beta/SparseReshape/Identity*
T0	*#
_output_shapes
:         
v
4linear/linear_model/beta/weighted_sum/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
ф
2linear/linear_model/beta/weighted_sum/GreaterEqualGreaterEqual<linear/linear_model/beta/weighted_sum/SparseReshape/Identity4linear/linear_model/beta/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:         
С
+linear/linear_model/beta/weighted_sum/WhereWhere2linear/linear_model/beta/weighted_sum/GreaterEqual*'
_output_shapes
:         
Ж
3linear/linear_model/beta/weighted_sum/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╓
-linear/linear_model/beta/weighted_sum/ReshapeReshape+linear/linear_model/beta/weighted_sum/Where3linear/linear_model/beta/weighted_sum/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
№
.linear/linear_model/beta/weighted_sum/Gather_1Gather3linear/linear_model/beta/weighted_sum/SparseReshape-linear/linear_model/beta/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*'
_output_shapes
:         
Б
.linear/linear_model/beta/weighted_sum/Gather_2Gather<linear/linear_model/beta/weighted_sum/SparseReshape/Identity-linear/linear_model/beta/weighted_sum/Reshape*
validate_indices(*
Tparams0	*
Tindices0	*#
_output_shapes
:         
Ц
.linear/linear_model/beta/weighted_sum/IdentityIdentity5linear/linear_model/beta/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
Б
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
Ч
Mlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
К
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_sliceStridedSlice.linear/linear_model/beta/weighted_sum/IdentityMlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stackOlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_1Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
┐
>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/CastCastGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice*

DstT0*

SrcT0	*
_output_shapes
: 
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╟
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/rangeRangeElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/start>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/CastElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range/delta*

Tidx0*#
_output_shapes
:         
╞
@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Cast_1Cast?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/range*

DstT0	*

SrcT0*#
_output_shapes
:         
а
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackConst*
dtype0*
valueB"        *
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Я
Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1StridedSlice.linear/linear_model/beta/weighted_sum/Gather_1Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stackQlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_1Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
з
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffListDiff@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Cast_1Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_1*
out_idx0*
T0	*2
_output_shapes 
:         :         
Щ
Olinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ы
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ы
Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Т
Ilinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2StridedSlice.linear/linear_model/beta/weighted_sum/IdentityOlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stackQlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_1Qlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0	*
shrink_axis_mask
У
Hlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
Ш
Dlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims
ExpandDimsIlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/strided_slice_2Hlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDims/dim*

Tdim0*
T0	*
_output_shapes
:
Ч
Ulinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ч
Ulinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/default_valueConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
ц
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDenseSparseToDenseBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffDlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ExpandDimsUlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/sparse_valuesUlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense/default_value*
validate_indices(*
Tindices0	*
T0
*#
_output_shapes
:         
Ш
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Щ
Alinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ReshapeReshapeBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiffGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape/shape*
Tshape0*
T0	*'
_output_shapes
:         
╞
Dlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/zeros_like	ZerosLikeAlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Reshape*
T0	*'
_output_shapes
:         
З
Elinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
у
@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concatConcatV2Alinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ReshapeDlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/zeros_likeElinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
┴
?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ShapeShapeBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/ListDiff*
out_type0*
T0	*
_output_shapes
:
Ў
>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/FillFill?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Shape?linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/Const*
T0	*#
_output_shapes
:         
Й
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╨
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1ConcatV2.linear/linear_model/beta/weighted_sum/Gather_1@linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concatGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1/axis*'
_output_shapes
:         *

Tidx0*
T0	*
N
Й
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╩
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2ConcatV2.linear/linear_model/beta/weighted_sum/Gather_2>linear/linear_model/beta/weighted_sum/SparseFillEmptyRows/FillGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2/axis*#
_output_shapes
:         *

Tidx0*
T0	*
N
╤
Glinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorderSparseReorderBlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_1Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/concat_2.linear/linear_model/beta/weighted_sum/Identity*
T0	*6
_output_shapes$
":         :         
г
Blinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/IdentityIdentity.linear/linear_model/beta/weighted_sum/Identity*
T0	*
_output_shapes
:
в
Qlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
д
Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
д
Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
└
Klinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorderQlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stackSlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Slinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
Index0*#
_output_shapes
:         *

begin_mask*
ellipsis_mask *
end_mask*
T0	*
shrink_axis_mask
╘
Blinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/CastCastKlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:         
х
Dlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/UniqueUniqueIlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseReorder:1*
out_idx0*
T0	*2
_output_shapes 
:         :         
ш
Nlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/embedding_lookupGather,linear/linear_model/beta/weights/part_0/readDlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Unique*
validate_indices(*
Tparams0*
Tindices0	*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*'
_output_shapes
:         
ы
=linear/linear_model/beta/weighted_sum/embedding_lookup_sparseSparseSegmentSumNlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/embedding_lookupFlinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Unique:1Blinear/linear_model/beta/weighted_sum/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:         
Ж
5linear/linear_model/beta/weighted_sum/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
·
/linear/linear_model/beta/weighted_sum/Reshape_1ReshapeGlinear/linear_model/beta/weighted_sum/SparseFillEmptyRows/SparseToDense5linear/linear_model/beta/weighted_sum/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
и
+linear/linear_model/beta/weighted_sum/ShapeShape=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Г
9linear/linear_model/beta/weighted_sum/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
Е
;linear/linear_model/beta/weighted_sum/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Е
;linear/linear_model/beta/weighted_sum/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╖
3linear/linear_model/beta/weighted_sum/strided_sliceStridedSlice+linear/linear_model/beta/weighted_sum/Shape9linear/linear_model/beta/weighted_sum/strided_slice/stack;linear/linear_model/beta/weighted_sum/strided_slice/stack_1;linear/linear_model/beta/weighted_sum/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
o
-linear/linear_model/beta/weighted_sum/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╤
+linear/linear_model/beta/weighted_sum/stackPack-linear/linear_model/beta/weighted_sum/stack/03linear/linear_model/beta/weighted_sum/strided_slice*
_output_shapes
:*

axis *
T0*
N
▌
*linear/linear_model/beta/weighted_sum/TileTile/linear/linear_model/beta/weighted_sum/Reshape_1+linear/linear_model/beta/weighted_sum/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
о
0linear/linear_model/beta/weighted_sum/zeros_like	ZerosLike=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
■
%linear/linear_model/beta/weighted_sumSelect*linear/linear_model/beta/weighted_sum/Tile0linear/linear_model/beta/weighted_sum/zeros_like=linear/linear_model/beta/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:         
Т
,linear/linear_model/beta/weighted_sum/Cast_1Cast(linear/linear_model/beta/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
}
3linear/linear_model/beta/weighted_sum/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
|
2linear/linear_model/beta/weighted_sum/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
 
-linear/linear_model/beta/weighted_sum/Slice_1Slice,linear/linear_model/beta/weighted_sum/Cast_13linear/linear_model/beta/weighted_sum/Slice_1/begin2linear/linear_model/beta/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
Т
-linear/linear_model/beta/weighted_sum/Shape_1Shape%linear/linear_model/beta/weighted_sum*
out_type0*
T0*
_output_shapes
:
}
3linear/linear_model/beta/weighted_sum/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Е
2linear/linear_model/beta/weighted_sum/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
А
-linear/linear_model/beta/weighted_sum/Slice_2Slice-linear/linear_model/beta/weighted_sum/Shape_13linear/linear_model/beta/weighted_sum/Slice_2/begin2linear/linear_model/beta/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
s
1linear/linear_model/beta/weighted_sum/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
Г
,linear/linear_model/beta/weighted_sum/concatConcatV2-linear/linear_model/beta/weighted_sum/Slice_1-linear/linear_model/beta/weighted_sum/Slice_21linear/linear_model/beta/weighted_sum/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╧
/linear/linear_model/beta/weighted_sum/Reshape_2Reshape%linear/linear_model/beta/weighted_sum,linear/linear_model/beta/weighted_sum/concat*
Tshape0*
T0*'
_output_shapes
:         
╬
(linear/linear_model/weighted_sum_no_biasAddN0linear/linear_model/alpha/weighted_sum/Reshape_2/linear/linear_model/beta/weighted_sum/Reshape_2*
N*
T0*'
_output_shapes
:         
┬
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
valueB*    *
_output_shapes
:
╧
'linear/linear_model/bias_weights/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
shared_name 
ж
.linear/linear_model/bias_weights/part_0/AssignAssign'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
validate_shape(*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
use_locking(*
T0*
_output_shapes
:
┬
,linear/linear_model/bias_weights/part_0/readIdentity'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
T0*
_output_shapes
:

 linear/linear_model/bias_weightsIdentity,linear/linear_model/bias_weights/part_0/read*
T0*
_output_shapes
:
└
 linear/linear_model/weighted_sumBiasAdd(linear/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
data_formatNHWC*
T0*'
_output_shapes
:         
^
linear/zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Т
linear/zero_fraction/EqualEqual linear/linear_model/weighted_sumlinear/zero_fraction/zero*
T0*'
_output_shapes
:         
~
linear/zero_fraction/CastCastlinear/zero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:         
k
linear/zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Ц
linear/zero_fraction/MeanMeanlinear/zero_fraction/Castlinear/zero_fraction/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Р
*linear/linear/fraction_of_zero_values/tagsConst*
dtype0*6
value-B+ B%linear/linear/fraction_of_zero_values*
_output_shapes
: 
Ю
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction/Mean*
T0*
_output_shapes
: 
u
linear/linear/activation/tagConst*
dtype0*)
value B Blinear/linear/activation*
_output_shapes
: 
Н
linear/linear/activationHistogramSummarylinear/linear/activation/tag linear/linear_model/weighted_sum*
T0*
_output_shapes
: 
r
addAdddnn/logits/BiasAdd linear/linear_model/weighted_sum*
T0*'
_output_shapes
:         
o
+binary_logistic_head/predictions/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:         
n
,binary_logistic_head/predictions/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
┌
'binary_logistic_head/predictions/concatConcatV2+binary_logistic_head/predictions/zeros_likeadd,binary_logistic_head/predictions/concat/axis*'
_output_shapes
:         *

Tidx0*
T0*
N
k
)binary_logistic_head/predictions/logisticSigmoidadd*
T0*'
_output_shapes
:         
Ф
.binary_logistic_head/predictions/probabilitiesSoftmax'binary_logistic_head/predictions/concat*
T0*'
_output_shapes
:         
t
2binary_logistic_head/predictions/classes/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
╔
(binary_logistic_head/predictions/classesArgMax'binary_logistic_head/predictions/concat2binary_logistic_head/predictions/classes/dimension*

Tidx0*
T0*#
_output_shapes
:         
Т
6binary_logistic_head/log_loss_with_two_classes/ToFloatCasthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
}
9binary_logistic_head/log_loss_with_two_classes/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:         
╜
;binary_logistic_head/log_loss_with_two_classes/GreaterEqualGreaterEqualadd9binary_logistic_head/log_loss_with_two_classes/zeros_like*
T0*'
_output_shapes
:         
ю
5binary_logistic_head/log_loss_with_two_classes/SelectSelect;binary_logistic_head/log_loss_with_two_classes/GreaterEqualadd9binary_logistic_head/log_loss_with_two_classes/zeros_like*
T0*'
_output_shapes
:         
p
2binary_logistic_head/log_loss_with_two_classes/NegNegadd*
T0*'
_output_shapes
:         
щ
7binary_logistic_head/log_loss_with_two_classes/Select_1Select;binary_logistic_head/log_loss_with_two_classes/GreaterEqual2binary_logistic_head/log_loss_with_two_classes/Negadd*
T0*'
_output_shapes
:         
и
2binary_logistic_head/log_loss_with_two_classes/mulMuladd6binary_logistic_head/log_loss_with_two_classes/ToFloat*
T0*'
_output_shapes
:         
╓
2binary_logistic_head/log_loss_with_two_classes/subSub5binary_logistic_head/log_loss_with_two_classes/Select2binary_logistic_head/log_loss_with_two_classes/mul*
T0*'
_output_shapes
:         
д
2binary_logistic_head/log_loss_with_two_classes/ExpExp7binary_logistic_head/log_loss_with_two_classes/Select_1*
T0*'
_output_shapes
:         
г
4binary_logistic_head/log_loss_with_two_classes/Log1pLog1p2binary_logistic_head/log_loss_with_two_classes/Exp*
T0*'
_output_shapes
:         
╤
.binary_logistic_head/log_loss_with_two_classesAdd2binary_logistic_head/log_loss_with_two_classes/sub4binary_logistic_head/log_loss_with_two_classes/Log1p*
T0*'
_output_shapes
:         
К
9binary_logistic_head/log_loss_with_two_classes/loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
ф
3binary_logistic_head/log_loss_with_two_classes/lossMean.binary_logistic_head/log_loss_with_two_classes9binary_logistic_head/log_loss_with_two_classes/loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
l
'binary_logistic_head/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
▓
"binary_logistic_head/ScalarSummaryScalarSummary'binary_logistic_head/ScalarSummary/tags3binary_logistic_head/log_loss_with_two_classes/loss*
T0*
_output_shapes
: 
l
'binary_logistic_head/metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
'binary_logistic_head/metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Р
.binary_logistic_head/metrics/mean/total/AssignAssign'binary_logistic_head/metrics/mean/total'binary_logistic_head/metrics/mean/zeros*
validate_shape(*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 
╛
,binary_logistic_head/metrics/mean/total/readIdentity'binary_logistic_head/metrics/mean/total*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Л
'binary_logistic_head/metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Т
.binary_logistic_head/metrics/mean/count/AssignAssign'binary_logistic_head/metrics/mean/count)binary_logistic_head/metrics/mean/zeros_1*
validate_shape(*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 
╛
,binary_logistic_head/metrics/mean/count/readIdentity'binary_logistic_head/metrics/mean/count*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
T0*
_output_shapes
: 
h
&binary_logistic_head/metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
Л
+binary_logistic_head/metrics/mean/ToFloat_1Cast&binary_logistic_head/metrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
j
'binary_logistic_head/metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
╚
%binary_logistic_head/metrics/mean/SumSum3binary_logistic_head/log_loss_with_two_classes/loss'binary_logistic_head/metrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
°
+binary_logistic_head/metrics/mean/AssignAdd	AssignAdd'binary_logistic_head/metrics/mean/total%binary_logistic_head/metrics/mean/Sum*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
╢
-binary_logistic_head/metrics/mean/AssignAdd_1	AssignAdd'binary_logistic_head/metrics/mean/count+binary_logistic_head/metrics/mean/ToFloat_14^binary_logistic_head/log_loss_with_two_classes/loss*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
░
)binary_logistic_head/metrics/mean/GreaterGreater,binary_logistic_head/metrics/mean/count/read+binary_logistic_head/metrics/mean/Greater/y*
T0*
_output_shapes
: 
▒
)binary_logistic_head/metrics/mean/truedivRealDiv,binary_logistic_head/metrics/mean/total/read,binary_logistic_head/metrics/mean/count/read*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╙
'binary_logistic_head/metrics/mean/valueSelect)binary_logistic_head/metrics/mean/Greater)binary_logistic_head/metrics/mean/truediv)binary_logistic_head/metrics/mean/value/e*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╡
+binary_logistic_head/metrics/mean/Greater_1Greater-binary_logistic_head/metrics/mean/AssignAdd_1-binary_logistic_head/metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
│
+binary_logistic_head/metrics/mean/truediv_1RealDiv+binary_logistic_head/metrics/mean/AssignAdd-binary_logistic_head/metrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
▀
+binary_logistic_head/metrics/mean/update_opSelect+binary_logistic_head/metrics/mean/Greater_1+binary_logistic_head/metrics/mean/truediv_1-binary_logistic_head/metrics/mean/update_op/e*
T0*
_output_shapes
: 
н
Abinary_logistic_head/metrics/remove_squeezable_dimensions/SqueezeSqueezehash_table_Lookup*
squeeze_dims

         *
T0	*#
_output_shapes
:         
╞
"binary_logistic_head/metrics/EqualEqual(binary_logistic_head/predictions/classesAbinary_logistic_head/metrics/remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
Н
$binary_logistic_head/metrics/ToFloatCast"binary_logistic_head/metrics/Equal*

DstT0*

SrcT0
*#
_output_shapes
:         
p
+binary_logistic_head/metrics/accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
П
+binary_logistic_head/metrics/accuracy/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
а
2binary_logistic_head/metrics/accuracy/total/AssignAssign+binary_logistic_head/metrics/accuracy/total+binary_logistic_head/metrics/accuracy/zeros*
validate_shape(*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
use_locking(*
T0*
_output_shapes
: 
╩
0binary_logistic_head/metrics/accuracy/total/readIdentity+binary_logistic_head/metrics/accuracy/total*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
П
+binary_logistic_head/metrics/accuracy/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
в
2binary_logistic_head/metrics/accuracy/count/AssignAssign+binary_logistic_head/metrics/accuracy/count-binary_logistic_head/metrics/accuracy/zeros_1*
validate_shape(*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
use_locking(*
T0*
_output_shapes
: 
╩
0binary_logistic_head/metrics/accuracy/count/readIdentity+binary_logistic_head/metrics/accuracy/count*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
T0*
_output_shapes
: 
Й
*binary_logistic_head/metrics/accuracy/SizeSize$binary_logistic_head/metrics/ToFloat*
out_type0*
T0*
_output_shapes
: 
У
/binary_logistic_head/metrics/accuracy/ToFloat_1Cast*binary_logistic_head/metrics/accuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
u
+binary_logistic_head/metrics/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
┴
)binary_logistic_head/metrics/accuracy/SumSum$binary_logistic_head/metrics/ToFloat+binary_logistic_head/metrics/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
И
/binary_logistic_head/metrics/accuracy/AssignAdd	AssignAdd+binary_logistic_head/metrics/accuracy/total)binary_logistic_head/metrics/accuracy/Sum*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
use_locking( *
T0*
_output_shapes
: 
╖
1binary_logistic_head/metrics/accuracy/AssignAdd_1	AssignAdd+binary_logistic_head/metrics/accuracy/count/binary_logistic_head/metrics/accuracy/ToFloat_1%^binary_logistic_head/metrics/ToFloat*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
use_locking( *
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╝
-binary_logistic_head/metrics/accuracy/GreaterGreater0binary_logistic_head/metrics/accuracy/count/read/binary_logistic_head/metrics/accuracy/Greater/y*
T0*
_output_shapes
: 
╜
-binary_logistic_head/metrics/accuracy/truedivRealDiv0binary_logistic_head/metrics/accuracy/total/read0binary_logistic_head/metrics/accuracy/count/read*
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
у
+binary_logistic_head/metrics/accuracy/valueSelect-binary_logistic_head/metrics/accuracy/Greater-binary_logistic_head/metrics/accuracy/truediv-binary_logistic_head/metrics/accuracy/value/e*
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┴
/binary_logistic_head/metrics/accuracy/Greater_1Greater1binary_logistic_head/metrics/accuracy/AssignAdd_11binary_logistic_head/metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
┐
/binary_logistic_head/metrics/accuracy/truediv_1RealDiv/binary_logistic_head/metrics/accuracy/AssignAdd1binary_logistic_head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
я
/binary_logistic_head/metrics/accuracy/update_opSelect/binary_logistic_head/metrics/accuracy/Greater_1/binary_logistic_head/metrics/accuracy/truediv_11binary_logistic_head/metrics/accuracy/update_op/e*
T0*
_output_shapes
: 
n
)binary_logistic_head/metrics/mean_1/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_1/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_1/total/AssignAssign)binary_logistic_head/metrics/mean_1/total)binary_logistic_head/metrics/mean_1/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_1/total/readIdentity)binary_logistic_head/metrics/mean_1/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_1/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_1/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_1/count/AssignAssign)binary_logistic_head/metrics/mean_1/count+binary_logistic_head/metrics/mean_1/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_1/count/readIdentity)binary_logistic_head/metrics/mean_1/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
T0*
_output_shapes
: 
М
(binary_logistic_head/metrics/mean_1/SizeSize)binary_logistic_head/predictions/logistic*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_1/ToFloat_1Cast(binary_logistic_head/metrics/mean_1/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┬
'binary_logistic_head/metrics/mean_1/SumSum)binary_logistic_head/predictions/logistic)binary_logistic_head/metrics/mean_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
А
-binary_logistic_head/metrics/mean_1/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_1/total'binary_logistic_head/metrics/mean_1/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
use_locking( *
T0*
_output_shapes
: 
┤
/binary_logistic_head/metrics/mean_1/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_1/count-binary_logistic_head/metrics/mean_1/ToFloat_1*^binary_logistic_head/predictions/logistic*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_1/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_1/GreaterGreater.binary_logistic_head/metrics/mean_1/count/read-binary_logistic_head/metrics/mean_1/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_1/truedivRealDiv.binary_logistic_head/metrics/mean_1/total/read.binary_logistic_head/metrics/mean_1/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_1/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_1/valueSelect+binary_logistic_head/metrics/mean_1/Greater+binary_logistic_head/metrics/mean_1/truediv+binary_logistic_head/metrics/mean_1/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_1/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_1/Greater_1Greater/binary_logistic_head/metrics/mean_1/AssignAdd_1/binary_logistic_head/metrics/mean_1/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_1/truediv_1RealDiv-binary_logistic_head/metrics/mean_1/AssignAdd/binary_logistic_head/metrics/mean_1/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_1/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_1/update_opSelect-binary_logistic_head/metrics/mean_1/Greater_1-binary_logistic_head/metrics/mean_1/truediv_1/binary_logistic_head/metrics/mean_1/update_op/e*
T0*
_output_shapes
: 
В
&binary_logistic_head/metrics/ToFloat_2Casthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
n
)binary_logistic_head/metrics/mean_2/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_2/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_2/total/AssignAssign)binary_logistic_head/metrics/mean_2/total)binary_logistic_head/metrics/mean_2/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_2/total/readIdentity)binary_logistic_head/metrics/mean_2/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_2/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_2/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_2/count/AssignAssign)binary_logistic_head/metrics/mean_2/count+binary_logistic_head/metrics/mean_2/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_2/count/readIdentity)binary_logistic_head/metrics/mean_2/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
T0*
_output_shapes
: 
Й
(binary_logistic_head/metrics/mean_2/SizeSize&binary_logistic_head/metrics/ToFloat_2*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_2/ToFloat_1Cast(binary_logistic_head/metrics/mean_2/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┐
'binary_logistic_head/metrics/mean_2/SumSum&binary_logistic_head/metrics/ToFloat_2)binary_logistic_head/metrics/mean_2/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
А
-binary_logistic_head/metrics/mean_2/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_2/total'binary_logistic_head/metrics/mean_2/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
use_locking( *
T0*
_output_shapes
: 
▒
/binary_logistic_head/metrics/mean_2/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_2/count-binary_logistic_head/metrics/mean_2/ToFloat_1'^binary_logistic_head/metrics/ToFloat_2*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_2/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_2/GreaterGreater.binary_logistic_head/metrics/mean_2/count/read-binary_logistic_head/metrics/mean_2/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_2/truedivRealDiv.binary_logistic_head/metrics/mean_2/total/read.binary_logistic_head/metrics/mean_2/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_2/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_2/valueSelect+binary_logistic_head/metrics/mean_2/Greater+binary_logistic_head/metrics/mean_2/truediv+binary_logistic_head/metrics/mean_2/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_2/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_2/Greater_1Greater/binary_logistic_head/metrics/mean_2/AssignAdd_1/binary_logistic_head/metrics/mean_2/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_2/truediv_1RealDiv-binary_logistic_head/metrics/mean_2/AssignAdd/binary_logistic_head/metrics/mean_2/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_2/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_2/update_opSelect-binary_logistic_head/metrics/mean_2/Greater_1-binary_logistic_head/metrics/mean_2/truediv_1/binary_logistic_head/metrics/mean_2/update_op/e*
T0*
_output_shapes
: 
В
&binary_logistic_head/metrics/ToFloat_3Casthash_table_Lookup*

DstT0*

SrcT0	*'
_output_shapes
:         
n
)binary_logistic_head/metrics/mean_3/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_3/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ш
0binary_logistic_head/metrics/mean_3/total/AssignAssign)binary_logistic_head/metrics/mean_3/total)binary_logistic_head/metrics/mean_3/zeros*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_3/total/readIdentity)binary_logistic_head/metrics/mean_3/total*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_3/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
Н
)binary_logistic_head/metrics/mean_3/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Ъ
0binary_logistic_head/metrics/mean_3/count/AssignAssign)binary_logistic_head/metrics/mean_3/count+binary_logistic_head/metrics/mean_3/zeros_1*
validate_shape(*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
use_locking(*
T0*
_output_shapes
: 
─
.binary_logistic_head/metrics/mean_3/count/readIdentity)binary_logistic_head/metrics/mean_3/count*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
T0*
_output_shapes
: 
Й
(binary_logistic_head/metrics/mean_3/SizeSize&binary_logistic_head/metrics/ToFloat_3*
out_type0*
T0*
_output_shapes
: 
П
-binary_logistic_head/metrics/mean_3/ToFloat_1Cast(binary_logistic_head/metrics/mean_3/Size*

DstT0*

SrcT0*
_output_shapes
: 
z
)binary_logistic_head/metrics/mean_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
┐
'binary_logistic_head/metrics/mean_3/SumSum&binary_logistic_head/metrics/ToFloat_3)binary_logistic_head/metrics/mean_3/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
А
-binary_logistic_head/metrics/mean_3/AssignAdd	AssignAdd)binary_logistic_head/metrics/mean_3/total'binary_logistic_head/metrics/mean_3/Sum*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
use_locking( *
T0*
_output_shapes
: 
▒
/binary_logistic_head/metrics/mean_3/AssignAdd_1	AssignAdd)binary_logistic_head/metrics/mean_3/count-binary_logistic_head/metrics/mean_3/ToFloat_1'^binary_logistic_head/metrics/ToFloat_3*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
use_locking( *
T0*
_output_shapes
: 
r
-binary_logistic_head/metrics/mean_3/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╢
+binary_logistic_head/metrics/mean_3/GreaterGreater.binary_logistic_head/metrics/mean_3/count/read-binary_logistic_head/metrics/mean_3/Greater/y*
T0*
_output_shapes
: 
╖
+binary_logistic_head/metrics/mean_3/truedivRealDiv.binary_logistic_head/metrics/mean_3/total/read.binary_logistic_head/metrics/mean_3/count/read*
T0*
_output_shapes
: 
p
+binary_logistic_head/metrics/mean_3/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
█
)binary_logistic_head/metrics/mean_3/valueSelect+binary_logistic_head/metrics/mean_3/Greater+binary_logistic_head/metrics/mean_3/truediv+binary_logistic_head/metrics/mean_3/value/e*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_3/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╗
-binary_logistic_head/metrics/mean_3/Greater_1Greater/binary_logistic_head/metrics/mean_3/AssignAdd_1/binary_logistic_head/metrics/mean_3/Greater_1/y*
T0*
_output_shapes
: 
╣
-binary_logistic_head/metrics/mean_3/truediv_1RealDiv-binary_logistic_head/metrics/mean_3/AssignAdd/binary_logistic_head/metrics/mean_3/AssignAdd_1*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/mean_3/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ч
-binary_logistic_head/metrics/mean_3/update_opSelect-binary_logistic_head/metrics/mean_3/Greater_1-binary_logistic_head/metrics/mean_3/truediv_1/binary_logistic_head/metrics/mean_3/update_op/e*
T0*
_output_shapes
: 
}
!binary_logistic_head/metrics/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         

.binary_logistic_head/metrics/auc/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╬
(binary_logistic_head/metrics/auc/ReshapeReshape)binary_logistic_head/predictions/logistic.binary_logistic_head/metrics/auc/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Б
0binary_logistic_head/metrics/auc/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╩
*binary_logistic_head/metrics/auc/Reshape_1Reshape!binary_logistic_head/metrics/Cast0binary_logistic_head/metrics/auc/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
О
&binary_logistic_head/metrics/auc/ShapeShape(binary_logistic_head/metrics/auc/Reshape*
out_type0*
T0*
_output_shapes
:
~
4binary_logistic_head/metrics/auc/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
А
6binary_logistic_head/metrics/auc/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
А
6binary_logistic_head/metrics/auc/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ю
.binary_logistic_head/metrics/auc/strided_sliceStridedSlice&binary_logistic_head/metrics/auc/Shape4binary_logistic_head/metrics/auc/strided_slice/stack6binary_logistic_head/metrics/auc/strided_slice/stack_16binary_logistic_head/metrics/auc/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Х
&binary_logistic_head/metrics/auc/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
y
/binary_logistic_head/metrics/auc/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
╚
+binary_logistic_head/metrics/auc/ExpandDims
ExpandDims&binary_logistic_head/metrics/auc/Const/binary_logistic_head/metrics/auc/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
j
(binary_logistic_head/metrics/auc/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
┬
&binary_logistic_head/metrics/auc/stackPack(binary_logistic_head/metrics/auc/stack/0.binary_logistic_head/metrics/auc/strided_slice*
_output_shapes
:*

axis *
T0*
N
╟
%binary_logistic_head/metrics/auc/TileTile+binary_logistic_head/metrics/auc/ExpandDims&binary_logistic_head/metrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
В
/binary_logistic_head/metrics/auc/transpose/RankRank(binary_logistic_head/metrics/auc/Reshape*
T0*
_output_shapes
: 
r
0binary_logistic_head/metrics/auc/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
╣
.binary_logistic_head/metrics/auc/transpose/subSub/binary_logistic_head/metrics/auc/transpose/Rank0binary_logistic_head/metrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
x
6binary_logistic_head/metrics/auc/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
x
6binary_logistic_head/metrics/auc/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
В
0binary_logistic_head/metrics/auc/transpose/RangeRange6binary_logistic_head/metrics/auc/transpose/Range/start/binary_logistic_head/metrics/auc/transpose/Rank6binary_logistic_head/metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:
╛
0binary_logistic_head/metrics/auc/transpose/sub_1Sub.binary_logistic_head/metrics/auc/transpose/sub0binary_logistic_head/metrics/auc/transpose/Range*
T0*
_output_shapes
:
╥
*binary_logistic_head/metrics/auc/transpose	Transpose(binary_logistic_head/metrics/auc/Reshape0binary_logistic_head/metrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
В
1binary_logistic_head/metrics/auc/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
╙
'binary_logistic_head/metrics/auc/Tile_1Tile*binary_logistic_head/metrics/auc/transpose1binary_logistic_head/metrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
╢
(binary_logistic_head/metrics/auc/GreaterGreater'binary_logistic_head/metrics/auc/Tile_1%binary_logistic_head/metrics/auc/Tile*
T0*(
_output_shapes
:╚         
Н
+binary_logistic_head/metrics/auc/LogicalNot
LogicalNot(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
В
1binary_logistic_head/metrics/auc/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
╙
'binary_logistic_head/metrics/auc/Tile_2Tile*binary_logistic_head/metrics/auc/Reshape_11binary_logistic_head/metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
О
-binary_logistic_head/metrics/auc/LogicalNot_1
LogicalNot'binary_logistic_head/metrics/auc/Tile_2*(
_output_shapes
:╚         
u
&binary_logistic_head/metrics/auc/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Э
/binary_logistic_head/metrics/auc/true_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
м
6binary_logistic_head/metrics/auc/true_positives/AssignAssign/binary_logistic_head/metrics/auc/true_positives&binary_logistic_head/metrics/auc/zeros*
validate_shape(*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
█
4binary_logistic_head/metrics/auc/true_positives/readIdentity/binary_logistic_head/metrics/auc/true_positives*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
T0*
_output_shapes	
:╚
╢
+binary_logistic_head/metrics/auc/LogicalAnd
LogicalAnd'binary_logistic_head/metrics/auc/Tile_2(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
б
*binary_logistic_head/metrics/auc/ToFloat_1Cast+binary_logistic_head/metrics/auc/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
x
6binary_logistic_head/metrics/auc/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╥
$binary_logistic_head/metrics/auc/SumSum*binary_logistic_head/metrics/auc/ToFloat_16binary_logistic_head/metrics/auc/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
Л
*binary_logistic_head/metrics/auc/AssignAdd	AssignAdd/binary_logistic_head/metrics/auc/true_positives$binary_logistic_head/metrics/auc/Sum*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Ю
0binary_logistic_head/metrics/auc/false_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
▒
7binary_logistic_head/metrics/auc/false_negatives/AssignAssign0binary_logistic_head/metrics/auc/false_negatives(binary_logistic_head/metrics/auc/zeros_1*
validate_shape(*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
▐
5binary_logistic_head/metrics/auc/false_negatives/readIdentity0binary_logistic_head/metrics/auc/false_negatives*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
T0*
_output_shapes	
:╚
╗
-binary_logistic_head/metrics/auc/LogicalAnd_1
LogicalAnd'binary_logistic_head/metrics/auc/Tile_2+binary_logistic_head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_2Cast-binary_logistic_head/metrics/auc/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_1Sum*binary_logistic_head/metrics/auc/ToFloat_28binary_logistic_head/metrics/auc/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
С
,binary_logistic_head/metrics/auc/AssignAdd_1	AssignAdd0binary_logistic_head/metrics/auc/false_negatives&binary_logistic_head/metrics/auc/Sum_1*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Э
/binary_logistic_head/metrics/auc/true_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
о
6binary_logistic_head/metrics/auc/true_negatives/AssignAssign/binary_logistic_head/metrics/auc/true_negatives(binary_logistic_head/metrics/auc/zeros_2*
validate_shape(*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
█
4binary_logistic_head/metrics/auc/true_negatives/readIdentity/binary_logistic_head/metrics/auc/true_negatives*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
T0*
_output_shapes	
:╚
┴
-binary_logistic_head/metrics/auc/LogicalAnd_2
LogicalAnd-binary_logistic_head/metrics/auc/LogicalNot_1+binary_logistic_head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_3Cast-binary_logistic_head/metrics/auc/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_2Sum*binary_logistic_head/metrics/auc/ToFloat_38binary_logistic_head/metrics/auc/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
П
,binary_logistic_head/metrics/auc/AssignAdd_2	AssignAdd/binary_logistic_head/metrics/auc/true_negatives&binary_logistic_head/metrics/auc/Sum_2*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
w
(binary_logistic_head/metrics/auc/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Ю
0binary_logistic_head/metrics/auc/false_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
▒
7binary_logistic_head/metrics/auc/false_positives/AssignAssign0binary_logistic_head/metrics/auc/false_positives(binary_logistic_head/metrics/auc/zeros_3*
validate_shape(*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
▐
5binary_logistic_head/metrics/auc/false_positives/readIdentity0binary_logistic_head/metrics/auc/false_positives*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
T0*
_output_shapes	
:╚
╛
-binary_logistic_head/metrics/auc/LogicalAnd_3
LogicalAnd-binary_logistic_head/metrics/auc/LogicalNot_1(binary_logistic_head/metrics/auc/Greater*(
_output_shapes
:╚         
г
*binary_logistic_head/metrics/auc/ToFloat_4Cast-binary_logistic_head/metrics/auc/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╓
&binary_logistic_head/metrics/auc/Sum_3Sum*binary_logistic_head/metrics/auc/ToFloat_48binary_logistic_head/metrics/auc/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
С
,binary_logistic_head/metrics/auc/AssignAdd_3	AssignAdd0binary_logistic_head/metrics/auc/false_positives&binary_logistic_head/metrics/auc/Sum_3*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
k
&binary_logistic_head/metrics/auc/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
п
$binary_logistic_head/metrics/auc/addAdd4binary_logistic_head/metrics/auc/true_positives/read&binary_logistic_head/metrics/auc/add/y*
T0*
_output_shapes	
:╚
└
&binary_logistic_head/metrics/auc/add_1Add4binary_logistic_head/metrics/auc/true_positives/read5binary_logistic_head/metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_2Add&binary_logistic_head/metrics/auc/add_1(binary_logistic_head/metrics/auc/add_2/y*
T0*
_output_shapes	
:╚
г
$binary_logistic_head/metrics/auc/divRealDiv$binary_logistic_head/metrics/auc/add&binary_logistic_head/metrics/auc/add_2*
T0*
_output_shapes	
:╚
└
&binary_logistic_head/metrics/auc/add_3Add5binary_logistic_head/metrics/auc/false_positives/read4binary_logistic_head/metrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_4/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_4Add&binary_logistic_head/metrics/auc/add_3(binary_logistic_head/metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
╢
&binary_logistic_head/metrics/auc/div_1RealDiv5binary_logistic_head/metrics/auc/false_positives/read&binary_logistic_head/metrics/auc/add_4*
T0*
_output_shapes	
:╚
А
6binary_logistic_head/metrics/auc/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_1StridedSlice&binary_logistic_head/metrics/auc/div_16binary_logistic_head/metrics/auc/strided_slice_1/stack8binary_logistic_head/metrics/auc/strided_slice_1/stack_18binary_logistic_head/metrics/auc/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_2StridedSlice&binary_logistic_head/metrics/auc/div_16binary_logistic_head/metrics/auc/strided_slice_2/stack8binary_logistic_head/metrics/auc/strided_slice_2/stack_18binary_logistic_head/metrics/auc/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╡
$binary_logistic_head/metrics/auc/subSub0binary_logistic_head/metrics/auc/strided_slice_10binary_logistic_head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
А
6binary_logistic_head/metrics/auc/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
й
0binary_logistic_head/metrics/auc/strided_slice_3StridedSlice$binary_logistic_head/metrics/auc/div6binary_logistic_head/metrics/auc/strided_slice_3/stack8binary_logistic_head/metrics/auc/strided_slice_3/stack_18binary_logistic_head/metrics/auc/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
й
0binary_logistic_head/metrics/auc/strided_slice_4StridedSlice$binary_logistic_head/metrics/auc/div6binary_logistic_head/metrics/auc/strided_slice_4/stack8binary_logistic_head/metrics/auc/strided_slice_4/stack_18binary_logistic_head/metrics/auc/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╖
&binary_logistic_head/metrics/auc/add_5Add0binary_logistic_head/metrics/auc/strided_slice_30binary_logistic_head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
o
*binary_logistic_head/metrics/auc/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
н
(binary_logistic_head/metrics/auc/truedivRealDiv&binary_logistic_head/metrics/auc/add_5*binary_logistic_head/metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
б
$binary_logistic_head/metrics/auc/MulMul$binary_logistic_head/metrics/auc/sub(binary_logistic_head/metrics/auc/truediv*
T0*
_output_shapes	
:╟
r
(binary_logistic_head/metrics/auc/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
╗
&binary_logistic_head/metrics/auc/valueSum$binary_logistic_head/metrics/auc/Mul(binary_logistic_head/metrics/auc/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
m
(binary_logistic_head/metrics/auc/add_6/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
й
&binary_logistic_head/metrics/auc/add_6Add*binary_logistic_head/metrics/auc/AssignAdd(binary_logistic_head/metrics/auc/add_6/y*
T0*
_output_shapes	
:╚
н
&binary_logistic_head/metrics/auc/add_7Add*binary_logistic_head/metrics/auc/AssignAdd,binary_logistic_head/metrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc/add_8/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
е
&binary_logistic_head/metrics/auc/add_8Add&binary_logistic_head/metrics/auc/add_7(binary_logistic_head/metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
з
&binary_logistic_head/metrics/auc/div_2RealDiv&binary_logistic_head/metrics/auc/add_6&binary_logistic_head/metrics/auc/add_8*
T0*
_output_shapes	
:╚
п
&binary_logistic_head/metrics/auc/add_9Add,binary_logistic_head/metrics/auc/AssignAdd_3,binary_logistic_head/metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
n
)binary_logistic_head/metrics/auc/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
з
'binary_logistic_head/metrics/auc/add_10Add&binary_logistic_head/metrics/auc/add_9)binary_logistic_head/metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
о
&binary_logistic_head/metrics/auc/div_3RealDiv,binary_logistic_head/metrics/auc/AssignAdd_3'binary_logistic_head/metrics/auc/add_10*
T0*
_output_shapes	
:╚
А
6binary_logistic_head/metrics/auc/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_5StridedSlice&binary_logistic_head/metrics/auc/div_36binary_logistic_head/metrics/auc/strided_slice_5/stack8binary_logistic_head/metrics/auc/strided_slice_5/stack_18binary_logistic_head/metrics/auc/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_6StridedSlice&binary_logistic_head/metrics/auc/div_36binary_logistic_head/metrics/auc/strided_slice_6/stack8binary_logistic_head/metrics/auc/strided_slice_6/stack_18binary_logistic_head/metrics/auc/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╖
&binary_logistic_head/metrics/auc/sub_1Sub0binary_logistic_head/metrics/auc/strided_slice_50binary_logistic_head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
А
6binary_logistic_head/metrics/auc/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
Г
8binary_logistic_head/metrics/auc/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_7StridedSlice&binary_logistic_head/metrics/auc/div_26binary_logistic_head/metrics/auc/strided_slice_7/stack8binary_logistic_head/metrics/auc/strided_slice_7/stack_18binary_logistic_head/metrics/auc/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
А
6binary_logistic_head/metrics/auc/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
л
0binary_logistic_head/metrics/auc/strided_slice_8StridedSlice&binary_logistic_head/metrics/auc/div_26binary_logistic_head/metrics/auc/strided_slice_8/stack8binary_logistic_head/metrics/auc/strided_slice_8/stack_18binary_logistic_head/metrics/auc/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╕
'binary_logistic_head/metrics/auc/add_11Add0binary_logistic_head/metrics/auc/strided_slice_70binary_logistic_head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
q
,binary_logistic_head/metrics/auc/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
▓
*binary_logistic_head/metrics/auc/truediv_1RealDiv'binary_logistic_head/metrics/auc/add_11,binary_logistic_head/metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
з
&binary_logistic_head/metrics/auc/Mul_1Mul&binary_logistic_head/metrics/auc/sub_1*binary_logistic_head/metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
r
(binary_logistic_head/metrics/auc/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
┴
*binary_logistic_head/metrics/auc/update_opSum&binary_logistic_head/metrics/auc/Mul_1(binary_logistic_head/metrics/auc/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

#binary_logistic_head/metrics/Cast_1Casthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
Б
0binary_logistic_head/metrics/auc_1/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╥
*binary_logistic_head/metrics/auc_1/ReshapeReshape)binary_logistic_head/predictions/logistic0binary_logistic_head/metrics/auc_1/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Г
2binary_logistic_head/metrics/auc_1/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
╨
,binary_logistic_head/metrics/auc_1/Reshape_1Reshape#binary_logistic_head/metrics/Cast_12binary_logistic_head/metrics/auc_1/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
Т
(binary_logistic_head/metrics/auc_1/ShapeShape*binary_logistic_head/metrics/auc_1/Reshape*
out_type0*
T0*
_output_shapes
:
А
6binary_logistic_head/metrics/auc_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
В
8binary_logistic_head/metrics/auc_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
В
8binary_logistic_head/metrics/auc_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
и
0binary_logistic_head/metrics/auc_1/strided_sliceStridedSlice(binary_logistic_head/metrics/auc_1/Shape6binary_logistic_head/metrics/auc_1/strided_slice/stack8binary_logistic_head/metrics/auc_1/strided_slice/stack_18binary_logistic_head/metrics/auc_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Ч
(binary_logistic_head/metrics/auc_1/ConstConst*
dtype0*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
_output_shapes	
:╚
{
1binary_logistic_head/metrics/auc_1/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
╬
-binary_logistic_head/metrics/auc_1/ExpandDims
ExpandDims(binary_logistic_head/metrics/auc_1/Const1binary_logistic_head/metrics/auc_1/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
l
*binary_logistic_head/metrics/auc_1/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╚
(binary_logistic_head/metrics/auc_1/stackPack*binary_logistic_head/metrics/auc_1/stack/00binary_logistic_head/metrics/auc_1/strided_slice*
_output_shapes
:*

axis *
T0*
N
═
'binary_logistic_head/metrics/auc_1/TileTile-binary_logistic_head/metrics/auc_1/ExpandDims(binary_logistic_head/metrics/auc_1/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
Ж
1binary_logistic_head/metrics/auc_1/transpose/RankRank*binary_logistic_head/metrics/auc_1/Reshape*
T0*
_output_shapes
: 
t
2binary_logistic_head/metrics/auc_1/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
┐
0binary_logistic_head/metrics/auc_1/transpose/subSub1binary_logistic_head/metrics/auc_1/transpose/Rank2binary_logistic_head/metrics/auc_1/transpose/sub/y*
T0*
_output_shapes
: 
z
8binary_logistic_head/metrics/auc_1/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
z
8binary_logistic_head/metrics/auc_1/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
К
2binary_logistic_head/metrics/auc_1/transpose/RangeRange8binary_logistic_head/metrics/auc_1/transpose/Range/start1binary_logistic_head/metrics/auc_1/transpose/Rank8binary_logistic_head/metrics/auc_1/transpose/Range/delta*

Tidx0*
_output_shapes
:
─
2binary_logistic_head/metrics/auc_1/transpose/sub_1Sub0binary_logistic_head/metrics/auc_1/transpose/sub2binary_logistic_head/metrics/auc_1/transpose/Range*
T0*
_output_shapes
:
╪
,binary_logistic_head/metrics/auc_1/transpose	Transpose*binary_logistic_head/metrics/auc_1/Reshape2binary_logistic_head/metrics/auc_1/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
Д
3binary_logistic_head/metrics/auc_1/Tile_1/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
┘
)binary_logistic_head/metrics/auc_1/Tile_1Tile,binary_logistic_head/metrics/auc_1/transpose3binary_logistic_head/metrics/auc_1/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
╝
*binary_logistic_head/metrics/auc_1/GreaterGreater)binary_logistic_head/metrics/auc_1/Tile_1'binary_logistic_head/metrics/auc_1/Tile*
T0*(
_output_shapes
:╚         
С
-binary_logistic_head/metrics/auc_1/LogicalNot
LogicalNot*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
Д
3binary_logistic_head/metrics/auc_1/Tile_2/multiplesConst*
dtype0*
valueB"╚      *
_output_shapes
:
┘
)binary_logistic_head/metrics/auc_1/Tile_2Tile,binary_logistic_head/metrics/auc_1/Reshape_13binary_logistic_head/metrics/auc_1/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
Т
/binary_logistic_head/metrics/auc_1/LogicalNot_1
LogicalNot)binary_logistic_head/metrics/auc_1/Tile_2*(
_output_shapes
:╚         
w
(binary_logistic_head/metrics/auc_1/zerosConst*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Я
1binary_logistic_head/metrics/auc_1/true_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
┤
8binary_logistic_head/metrics/auc_1/true_positives/AssignAssign1binary_logistic_head/metrics/auc_1/true_positives(binary_logistic_head/metrics/auc_1/zeros*
validate_shape(*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
use_locking(*
T0*
_output_shapes	
:╚
с
6binary_logistic_head/metrics/auc_1/true_positives/readIdentity1binary_logistic_head/metrics/auc_1/true_positives*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
T0*
_output_shapes	
:╚
╝
-binary_logistic_head/metrics/auc_1/LogicalAnd
LogicalAnd)binary_logistic_head/metrics/auc_1/Tile_2*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
е
,binary_logistic_head/metrics/auc_1/ToFloat_1Cast-binary_logistic_head/metrics/auc_1/LogicalAnd*

DstT0*

SrcT0
*(
_output_shapes
:╚         
z
8binary_logistic_head/metrics/auc_1/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
╪
&binary_logistic_head/metrics/auc_1/SumSum,binary_logistic_head/metrics/auc_1/ToFloat_18binary_logistic_head/metrics/auc_1/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
У
,binary_logistic_head/metrics/auc_1/AssignAdd	AssignAdd1binary_logistic_head/metrics/auc_1/true_positives&binary_logistic_head/metrics/auc_1/Sum*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_1Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
а
2binary_logistic_head/metrics/auc_1/false_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
╣
9binary_logistic_head/metrics/auc_1/false_negatives/AssignAssign2binary_logistic_head/metrics/auc_1/false_negatives*binary_logistic_head/metrics/auc_1/zeros_1*
validate_shape(*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
use_locking(*
T0*
_output_shapes	
:╚
ф
7binary_logistic_head/metrics/auc_1/false_negatives/readIdentity2binary_logistic_head/metrics/auc_1/false_negatives*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
T0*
_output_shapes	
:╚
┴
/binary_logistic_head/metrics/auc_1/LogicalAnd_1
LogicalAnd)binary_logistic_head/metrics/auc_1/Tile_2-binary_logistic_head/metrics/auc_1/LogicalNot*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_2Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_1*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_1Sum,binary_logistic_head/metrics/auc_1/ToFloat_2:binary_logistic_head/metrics/auc_1/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
Щ
.binary_logistic_head/metrics/auc_1/AssignAdd_1	AssignAdd2binary_logistic_head/metrics/auc_1/false_negatives(binary_logistic_head/metrics/auc_1/Sum_1*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_2Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
Я
1binary_logistic_head/metrics/auc_1/true_negatives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
╢
8binary_logistic_head/metrics/auc_1/true_negatives/AssignAssign1binary_logistic_head/metrics/auc_1/true_negatives*binary_logistic_head/metrics/auc_1/zeros_2*
validate_shape(*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
use_locking(*
T0*
_output_shapes	
:╚
с
6binary_logistic_head/metrics/auc_1/true_negatives/readIdentity1binary_logistic_head/metrics/auc_1/true_negatives*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
T0*
_output_shapes	
:╚
╟
/binary_logistic_head/metrics/auc_1/LogicalAnd_2
LogicalAnd/binary_logistic_head/metrics/auc_1/LogicalNot_1-binary_logistic_head/metrics/auc_1/LogicalNot*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_3Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_2*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_2/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_2Sum,binary_logistic_head/metrics/auc_1/ToFloat_3:binary_logistic_head/metrics/auc_1/Sum_2/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
Ч
.binary_logistic_head/metrics/auc_1/AssignAdd_2	AssignAdd1binary_logistic_head/metrics/auc_1/true_negatives(binary_logistic_head/metrics/auc_1/Sum_2*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
use_locking( *
T0*
_output_shapes	
:╚
y
*binary_logistic_head/metrics/auc_1/zeros_3Const*
dtype0*
valueB╚*    *
_output_shapes	
:╚
а
2binary_logistic_head/metrics/auc_1/false_positives
VariableV2*
dtype0*
shape:╚*
shared_name *
	container *
_output_shapes	
:╚
╣
9binary_logistic_head/metrics/auc_1/false_positives/AssignAssign2binary_logistic_head/metrics/auc_1/false_positives*binary_logistic_head/metrics/auc_1/zeros_3*
validate_shape(*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
use_locking(*
T0*
_output_shapes	
:╚
ф
7binary_logistic_head/metrics/auc_1/false_positives/readIdentity2binary_logistic_head/metrics/auc_1/false_positives*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
T0*
_output_shapes	
:╚
─
/binary_logistic_head/metrics/auc_1/LogicalAnd_3
LogicalAnd/binary_logistic_head/metrics/auc_1/LogicalNot_1*binary_logistic_head/metrics/auc_1/Greater*(
_output_shapes
:╚         
з
,binary_logistic_head/metrics/auc_1/ToFloat_4Cast/binary_logistic_head/metrics/auc_1/LogicalAnd_3*

DstT0*

SrcT0
*(
_output_shapes
:╚         
|
:binary_logistic_head/metrics/auc_1/Sum_3/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
▄
(binary_logistic_head/metrics/auc_1/Sum_3Sum,binary_logistic_head/metrics/auc_1/ToFloat_4:binary_logistic_head/metrics/auc_1/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:╚
Щ
.binary_logistic_head/metrics/auc_1/AssignAdd_3	AssignAdd2binary_logistic_head/metrics/auc_1/false_positives(binary_logistic_head/metrics/auc_1/Sum_3*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
use_locking( *
T0*
_output_shapes	
:╚
m
(binary_logistic_head/metrics/auc_1/add/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
╡
&binary_logistic_head/metrics/auc_1/addAdd6binary_logistic_head/metrics/auc_1/true_positives/read(binary_logistic_head/metrics/auc_1/add/y*
T0*
_output_shapes	
:╚
╞
(binary_logistic_head/metrics/auc_1/add_1Add6binary_logistic_head/metrics/auc_1/true_positives/read7binary_logistic_head/metrics/auc_1/false_negatives/read*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_2/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_2Add(binary_logistic_head/metrics/auc_1/add_1*binary_logistic_head/metrics/auc_1/add_2/y*
T0*
_output_shapes	
:╚
й
&binary_logistic_head/metrics/auc_1/divRealDiv&binary_logistic_head/metrics/auc_1/add(binary_logistic_head/metrics/auc_1/add_2*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_3/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
╣
(binary_logistic_head/metrics/auc_1/add_3Add6binary_logistic_head/metrics/auc_1/true_positives/read*binary_logistic_head/metrics/auc_1/add_3/y*
T0*
_output_shapes	
:╚
╞
(binary_logistic_head/metrics/auc_1/add_4Add6binary_logistic_head/metrics/auc_1/true_positives/read7binary_logistic_head/metrics/auc_1/false_positives/read*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_5/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_5Add(binary_logistic_head/metrics/auc_1/add_4*binary_logistic_head/metrics/auc_1/add_5/y*
T0*
_output_shapes	
:╚
н
(binary_logistic_head/metrics/auc_1/div_1RealDiv(binary_logistic_head/metrics/auc_1/add_3(binary_logistic_head/metrics/auc_1/add_5*
T0*
_output_shapes	
:╚
В
8binary_logistic_head/metrics/auc_1/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
│
2binary_logistic_head/metrics/auc_1/strided_slice_1StridedSlice&binary_logistic_head/metrics/auc_1/div8binary_logistic_head/metrics/auc_1/strided_slice_1/stack:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_2/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
│
2binary_logistic_head/metrics/auc_1/strided_slice_2StridedSlice&binary_logistic_head/metrics/auc_1/div8binary_logistic_head/metrics/auc_1/strided_slice_2/stack:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╗
&binary_logistic_head/metrics/auc_1/subSub2binary_logistic_head/metrics/auc_1/strided_slice_12binary_logistic_head/metrics/auc_1/strided_slice_2*
T0*
_output_shapes	
:╟
В
8binary_logistic_head/metrics/auc_1/strided_slice_3/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_3StridedSlice(binary_logistic_head/metrics/auc_1/div_18binary_logistic_head/metrics/auc_1/strided_slice_3/stack:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_4/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_4StridedSlice(binary_logistic_head/metrics/auc_1/div_18binary_logistic_head/metrics/auc_1/strided_slice_4/stack:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╜
(binary_logistic_head/metrics/auc_1/add_6Add2binary_logistic_head/metrics/auc_1/strided_slice_32binary_logistic_head/metrics/auc_1/strided_slice_4*
T0*
_output_shapes	
:╟
q
,binary_logistic_head/metrics/auc_1/truediv/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
│
*binary_logistic_head/metrics/auc_1/truedivRealDiv(binary_logistic_head/metrics/auc_1/add_6,binary_logistic_head/metrics/auc_1/truediv/y*
T0*
_output_shapes	
:╟
з
&binary_logistic_head/metrics/auc_1/MulMul&binary_logistic_head/metrics/auc_1/sub*binary_logistic_head/metrics/auc_1/truediv*
T0*
_output_shapes	
:╟
t
*binary_logistic_head/metrics/auc_1/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
┴
(binary_logistic_head/metrics/auc_1/valueSum&binary_logistic_head/metrics/auc_1/Mul*binary_logistic_head/metrics/auc_1/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
o
*binary_logistic_head/metrics/auc_1/add_7/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
п
(binary_logistic_head/metrics/auc_1/add_7Add,binary_logistic_head/metrics/auc_1/AssignAdd*binary_logistic_head/metrics/auc_1/add_7/y*
T0*
_output_shapes	
:╚
│
(binary_logistic_head/metrics/auc_1/add_8Add,binary_logistic_head/metrics/auc_1/AssignAdd.binary_logistic_head/metrics/auc_1/AssignAdd_1*
T0*
_output_shapes	
:╚
o
*binary_logistic_head/metrics/auc_1/add_9/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
л
(binary_logistic_head/metrics/auc_1/add_9Add(binary_logistic_head/metrics/auc_1/add_8*binary_logistic_head/metrics/auc_1/add_9/y*
T0*
_output_shapes	
:╚
н
(binary_logistic_head/metrics/auc_1/div_2RealDiv(binary_logistic_head/metrics/auc_1/add_7(binary_logistic_head/metrics/auc_1/add_9*
T0*
_output_shapes	
:╚
p
+binary_logistic_head/metrics/auc_1/add_10/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
▒
)binary_logistic_head/metrics/auc_1/add_10Add,binary_logistic_head/metrics/auc_1/AssignAdd+binary_logistic_head/metrics/auc_1/add_10/y*
T0*
_output_shapes	
:╚
┤
)binary_logistic_head/metrics/auc_1/add_11Add,binary_logistic_head/metrics/auc_1/AssignAdd.binary_logistic_head/metrics/auc_1/AssignAdd_3*
T0*
_output_shapes	
:╚
p
+binary_logistic_head/metrics/auc_1/add_12/yConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: 
о
)binary_logistic_head/metrics/auc_1/add_12Add)binary_logistic_head/metrics/auc_1/add_11+binary_logistic_head/metrics/auc_1/add_12/y*
T0*
_output_shapes	
:╚
п
(binary_logistic_head/metrics/auc_1/div_3RealDiv)binary_logistic_head/metrics/auc_1/add_10)binary_logistic_head/metrics/auc_1/add_12*
T0*
_output_shapes	
:╚
В
8binary_logistic_head/metrics/auc_1/strided_slice_5/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_5StridedSlice(binary_logistic_head/metrics/auc_1/div_28binary_logistic_head/metrics/auc_1/strided_slice_5/stack:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_6/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_6StridedSlice(binary_logistic_head/metrics/auc_1/div_28binary_logistic_head/metrics/auc_1/strided_slice_6/stack:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╜
(binary_logistic_head/metrics/auc_1/sub_1Sub2binary_logistic_head/metrics/auc_1/strided_slice_52binary_logistic_head/metrics/auc_1/strided_slice_6*
T0*
_output_shapes	
:╟
В
8binary_logistic_head/metrics/auc_1/strided_slice_7/stackConst*
dtype0*
valueB: *
_output_shapes
:
Е
:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_1Const*
dtype0*
valueB:╟*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_7StridedSlice(binary_logistic_head/metrics/auc_1/div_38binary_logistic_head/metrics/auc_1/strided_slice_7/stack:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
В
8binary_logistic_head/metrics/auc_1/strided_slice_8/stackConst*
dtype0*
valueB:*
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Д
:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╡
2binary_logistic_head/metrics/auc_1/strided_slice_8StridedSlice(binary_logistic_head/metrics/auc_1/div_38binary_logistic_head/metrics/auc_1/strided_slice_8/stack:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_1:binary_logistic_head/metrics/auc_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes	
:╟*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
╛
)binary_logistic_head/metrics/auc_1/add_13Add2binary_logistic_head/metrics/auc_1/strided_slice_72binary_logistic_head/metrics/auc_1/strided_slice_8*
T0*
_output_shapes	
:╟
s
.binary_logistic_head/metrics/auc_1/truediv_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
╕
,binary_logistic_head/metrics/auc_1/truediv_1RealDiv)binary_logistic_head/metrics/auc_1/add_13.binary_logistic_head/metrics/auc_1/truediv_1/y*
T0*
_output_shapes	
:╟
н
(binary_logistic_head/metrics/auc_1/Mul_1Mul(binary_logistic_head/metrics/auc_1/sub_1,binary_logistic_head/metrics/auc_1/truediv_1*
T0*
_output_shapes	
:╟
t
*binary_logistic_head/metrics/auc_1/Const_2Const*
dtype0*
valueB: *
_output_shapes
:
╟
,binary_logistic_head/metrics/auc_1/update_opSum(binary_logistic_head/metrics/auc_1/Mul_1*binary_logistic_head/metrics/auc_1/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
p
+binary_logistic_head/metrics/GreaterEqual/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
├
)binary_logistic_head/metrics/GreaterEqualGreaterEqual)binary_logistic_head/predictions/logistic+binary_logistic_head/metrics/GreaterEqual/y*
T0*'
_output_shapes
:         
Ъ
&binary_logistic_head/metrics/ToFloat_6Cast)binary_logistic_head/metrics/GreaterEqual*

DstT0*

SrcT0
*'
_output_shapes
:         
Ф
#binary_logistic_head/metrics/Cast_2Cast&binary_logistic_head/metrics/ToFloat_6*

DstT0	*

SrcT0*'
_output_shapes
:         
Ч
$binary_logistic_head/metrics/Equal_1Equal#binary_logistic_head/metrics/Cast_2hash_table_Lookup*
T0	*'
_output_shapes
:         
Х
&binary_logistic_head/metrics/ToFloat_7Cast$binary_logistic_head/metrics/Equal_1*

DstT0*

SrcT0
*'
_output_shapes
:         
r
-binary_logistic_head/metrics/accuracy_1/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
С
-binary_logistic_head/metrics/accuracy_1/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
и
4binary_logistic_head/metrics/accuracy_1/total/AssignAssign-binary_logistic_head/metrics/accuracy_1/total-binary_logistic_head/metrics/accuracy_1/zeros*
validate_shape(*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
use_locking(*
T0*
_output_shapes
: 
╨
2binary_logistic_head/metrics/accuracy_1/total/readIdentity-binary_logistic_head/metrics/accuracy_1/total*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy_1/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
С
-binary_logistic_head/metrics/accuracy_1/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
к
4binary_logistic_head/metrics/accuracy_1/count/AssignAssign-binary_logistic_head/metrics/accuracy_1/count/binary_logistic_head/metrics/accuracy_1/zeros_1*
validate_shape(*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
use_locking(*
T0*
_output_shapes
: 
╨
2binary_logistic_head/metrics/accuracy_1/count/readIdentity-binary_logistic_head/metrics/accuracy_1/count*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
T0*
_output_shapes
: 
Н
,binary_logistic_head/metrics/accuracy_1/SizeSize&binary_logistic_head/metrics/ToFloat_7*
out_type0*
T0*
_output_shapes
: 
Ч
1binary_logistic_head/metrics/accuracy_1/ToFloat_1Cast,binary_logistic_head/metrics/accuracy_1/Size*

DstT0*

SrcT0*
_output_shapes
: 
~
-binary_logistic_head/metrics/accuracy_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
╟
+binary_logistic_head/metrics/accuracy_1/SumSum&binary_logistic_head/metrics/ToFloat_7-binary_logistic_head/metrics/accuracy_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Р
1binary_logistic_head/metrics/accuracy_1/AssignAdd	AssignAdd-binary_logistic_head/metrics/accuracy_1/total+binary_logistic_head/metrics/accuracy_1/Sum*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
use_locking( *
T0*
_output_shapes
: 
┴
3binary_logistic_head/metrics/accuracy_1/AssignAdd_1	AssignAdd-binary_logistic_head/metrics/accuracy_1/count1binary_logistic_head/metrics/accuracy_1/ToFloat_1'^binary_logistic_head/metrics/ToFloat_7*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
use_locking( *
T0*
_output_shapes
: 
v
1binary_logistic_head/metrics/accuracy_1/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
┬
/binary_logistic_head/metrics/accuracy_1/GreaterGreater2binary_logistic_head/metrics/accuracy_1/count/read1binary_logistic_head/metrics/accuracy_1/Greater/y*
T0*
_output_shapes
: 
├
/binary_logistic_head/metrics/accuracy_1/truedivRealDiv2binary_logistic_head/metrics/accuracy_1/total/read2binary_logistic_head/metrics/accuracy_1/count/read*
T0*
_output_shapes
: 
t
/binary_logistic_head/metrics/accuracy_1/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ы
-binary_logistic_head/metrics/accuracy_1/valueSelect/binary_logistic_head/metrics/accuracy_1/Greater/binary_logistic_head/metrics/accuracy_1/truediv/binary_logistic_head/metrics/accuracy_1/value/e*
T0*
_output_shapes
: 
x
3binary_logistic_head/metrics/accuracy_1/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
╟
1binary_logistic_head/metrics/accuracy_1/Greater_1Greater3binary_logistic_head/metrics/accuracy_1/AssignAdd_13binary_logistic_head/metrics/accuracy_1/Greater_1/y*
T0*
_output_shapes
: 
┼
1binary_logistic_head/metrics/accuracy_1/truediv_1RealDiv1binary_logistic_head/metrics/accuracy_1/AssignAdd3binary_logistic_head/metrics/accuracy_1/AssignAdd_1*
T0*
_output_shapes
: 
x
3binary_logistic_head/metrics/accuracy_1/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ў
1binary_logistic_head/metrics/accuracy_1/update_opSelect1binary_logistic_head/metrics/accuracy_1/Greater_11binary_logistic_head/metrics/accuracy_1/truediv_13binary_logistic_head/metrics/accuracy_1/update_op/e*
T0*
_output_shapes
: 
Х
9binary_logistic_head/metrics/precision_at_thresholds/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/precision_at_thresholds/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ў
<binary_logistic_head/metrics/precision_at_thresholds/ReshapeReshape)binary_logistic_head/predictions/logisticBbinary_logistic_head/metrics/precision_at_thresholds/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Х
Dbinary_logistic_head/metrics/precision_at_thresholds/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
К
>binary_logistic_head/metrics/precision_at_thresholds/Reshape_1Reshape9binary_logistic_head/metrics/precision_at_thresholds/CastDbinary_logistic_head/metrics/precision_at_thresholds/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
╢
:binary_logistic_head/metrics/precision_at_thresholds/ShapeShape<binary_logistic_head/metrics/precision_at_thresholds/Reshape*
out_type0*
T0*
_output_shapes
:
Т
Hbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ф
Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ф
Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
В
Bbinary_logistic_head/metrics/precision_at_thresholds/strided_sliceStridedSlice:binary_logistic_head/metrics/precision_at_thresholds/ShapeHbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stackJbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_1Jbinary_logistic_head/metrics/precision_at_thresholds/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
З
:binary_logistic_head/metrics/precision_at_thresholds/ConstConst*
dtype0*
valueB*   ?*
_output_shapes
:
Н
Cbinary_logistic_head/metrics/precision_at_thresholds/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
Г
?binary_logistic_head/metrics/precision_at_thresholds/ExpandDims
ExpandDims:binary_logistic_head/metrics/precision_at_thresholds/ConstCbinary_logistic_head/metrics/precision_at_thresholds/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
~
<binary_logistic_head/metrics/precision_at_thresholds/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
■
:binary_logistic_head/metrics/precision_at_thresholds/stackPack<binary_logistic_head/metrics/precision_at_thresholds/stack/0Bbinary_logistic_head/metrics/precision_at_thresholds/strided_slice*
_output_shapes
:*

axis *
T0*
N
В
9binary_logistic_head/metrics/precision_at_thresholds/TileTile?binary_logistic_head/metrics/precision_at_thresholds/ExpandDims:binary_logistic_head/metrics/precision_at_thresholds/stack*

Tmultiples0*
T0*'
_output_shapes
:         
к
Cbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankRank<binary_logistic_head/metrics/precision_at_thresholds/Reshape*
T0*
_output_shapes
: 
Ж
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
ї
Bbinary_logistic_head/metrics/precision_at_thresholds/transpose/subSubCbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankDbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub/y*
T0*
_output_shapes
: 
М
Jbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
М
Jbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╥
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/RangeRangeJbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/startCbinary_logistic_head/metrics/precision_at_thresholds/transpose/RankJbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range/delta*

Tidx0*
_output_shapes
:
·
Dbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub_1SubBbinary_logistic_head/metrics/precision_at_thresholds/transpose/subDbinary_logistic_head/metrics/precision_at_thresholds/transpose/Range*
T0*
_output_shapes
:
О
>binary_logistic_head/metrics/precision_at_thresholds/transpose	Transpose<binary_logistic_head/metrics/precision_at_thresholds/ReshapeDbinary_logistic_head/metrics/precision_at_thresholds/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
Ц
Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_1/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
О
;binary_logistic_head/metrics/precision_at_thresholds/Tile_1Tile>binary_logistic_head/metrics/precision_at_thresholds/transposeEbinary_logistic_head/metrics/precision_at_thresholds/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
ё
<binary_logistic_head/metrics/precision_at_thresholds/GreaterGreater;binary_logistic_head/metrics/precision_at_thresholds/Tile_19binary_logistic_head/metrics/precision_at_thresholds/Tile*
T0*'
_output_shapes
:         
Ц
Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_2/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
О
;binary_logistic_head/metrics/precision_at_thresholds/Tile_2Tile>binary_logistic_head/metrics/precision_at_thresholds/Reshape_1Ebinary_logistic_head/metrics/precision_at_thresholds/Tile_2/multiples*

Tmultiples0*
T0
*'
_output_shapes
:         
│
?binary_logistic_head/metrics/precision_at_thresholds/LogicalNot
LogicalNot;binary_logistic_head/metrics/precision_at_thresholds/Tile_2*'
_output_shapes
:         
З
:binary_logistic_head/metrics/precision_at_thresholds/zerosConst*
dtype0*
valueB*    *
_output_shapes
:
п
Cbinary_logistic_head/metrics/precision_at_thresholds/true_positives
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
√
Jbinary_logistic_head/metrics/precision_at_thresholds/true_positives/AssignAssignCbinary_logistic_head/metrics/precision_at_thresholds/true_positives:binary_logistic_head/metrics/precision_at_thresholds/zeros*
validate_shape(*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
use_locking(*
T0*
_output_shapes
:
Ц
Hbinary_logistic_head/metrics/precision_at_thresholds/true_positives/readIdentityCbinary_logistic_head/metrics/precision_at_thresholds/true_positives*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
T0*
_output_shapes
:
ё
?binary_logistic_head/metrics/precision_at_thresholds/LogicalAnd
LogicalAnd;binary_logistic_head/metrics/precision_at_thresholds/Tile_2<binary_logistic_head/metrics/precision_at_thresholds/Greater*'
_output_shapes
:         
╚
>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_1Cast?binary_logistic_head/metrics/precision_at_thresholds/LogicalAnd*

DstT0*

SrcT0
*'
_output_shapes
:         
М
Jbinary_logistic_head/metrics/precision_at_thresholds/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Н
8binary_logistic_head/metrics/precision_at_thresholds/SumSum>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_1Jbinary_logistic_head/metrics/precision_at_thresholds/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
┌
>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd	AssignAddCbinary_logistic_head/metrics/precision_at_thresholds/true_positives8binary_logistic_head/metrics/precision_at_thresholds/Sum*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
use_locking( *
T0*
_output_shapes
:
Й
<binary_logistic_head/metrics/precision_at_thresholds/zeros_1Const*
dtype0*
valueB*    *
_output_shapes
:
░
Dbinary_logistic_head/metrics/precision_at_thresholds/false_positives
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
А
Kbinary_logistic_head/metrics/precision_at_thresholds/false_positives/AssignAssignDbinary_logistic_head/metrics/precision_at_thresholds/false_positives<binary_logistic_head/metrics/precision_at_thresholds/zeros_1*
validate_shape(*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
use_locking(*
T0*
_output_shapes
:
Щ
Ibinary_logistic_head/metrics/precision_at_thresholds/false_positives/readIdentityDbinary_logistic_head/metrics/precision_at_thresholds/false_positives*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
T0*
_output_shapes
:
ў
Abinary_logistic_head/metrics/precision_at_thresholds/LogicalAnd_1
LogicalAnd?binary_logistic_head/metrics/precision_at_thresholds/LogicalNot<binary_logistic_head/metrics/precision_at_thresholds/Greater*'
_output_shapes
:         
╩
>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_2CastAbinary_logistic_head/metrics/precision_at_thresholds/LogicalAnd_1*

DstT0*

SrcT0
*'
_output_shapes
:         
О
Lbinary_logistic_head/metrics/precision_at_thresholds/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
С
:binary_logistic_head/metrics/precision_at_thresholds/Sum_1Sum>binary_logistic_head/metrics/precision_at_thresholds/ToFloat_2Lbinary_logistic_head/metrics/precision_at_thresholds/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
@binary_logistic_head/metrics/precision_at_thresholds/AssignAdd_1	AssignAddDbinary_logistic_head/metrics/precision_at_thresholds/false_positives:binary_logistic_head/metrics/precision_at_thresholds/Sum_1*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
use_locking( *
T0*
_output_shapes
:

:binary_logistic_head/metrics/precision_at_thresholds/add/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
ъ
8binary_logistic_head/metrics/precision_at_thresholds/addAdd:binary_logistic_head/metrics/precision_at_thresholds/add/xHbinary_logistic_head/metrics/precision_at_thresholds/true_positives/read*
T0*
_output_shapes
:
ы
:binary_logistic_head/metrics/precision_at_thresholds/add_1Add8binary_logistic_head/metrics/precision_at_thresholds/addIbinary_logistic_head/metrics/precision_at_thresholds/false_positives/read*
T0*
_output_shapes
:
·
Dbinary_logistic_head/metrics/precision_at_thresholds/precision_valueRealDivHbinary_logistic_head/metrics/precision_at_thresholds/true_positives/read:binary_logistic_head/metrics/precision_at_thresholds/add_1*
T0*
_output_shapes
:
Б
<binary_logistic_head/metrics/precision_at_thresholds/add_2/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
ф
:binary_logistic_head/metrics/precision_at_thresholds/add_2Add<binary_logistic_head/metrics/precision_at_thresholds/add_2/x>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd*
T0*
_output_shapes
:
ф
:binary_logistic_head/metrics/precision_at_thresholds/add_3Add:binary_logistic_head/metrics/precision_at_thresholds/add_2@binary_logistic_head/metrics/precision_at_thresholds/AssignAdd_1*
T0*
_output_shapes
:
Ї
Hbinary_logistic_head/metrics/precision_at_thresholds/precision_update_opRealDiv>binary_logistic_head/metrics/precision_at_thresholds/AssignAdd:binary_logistic_head/metrics/precision_at_thresholds/add_3*
T0*
_output_shapes
:
к
$binary_logistic_head/metrics/SqueezeSqueezeDbinary_logistic_head/metrics/precision_at_thresholds/precision_value*
squeeze_dims
 *
T0*
_output_shapes
: 
░
&binary_logistic_head/metrics/Squeeze_1SqueezeHbinary_logistic_head/metrics/precision_at_thresholds/precision_update_op*
squeeze_dims
 *
T0*
_output_shapes
: 
Т
6binary_logistic_head/metrics/recall_at_thresholds/CastCasthash_table_Lookup*

DstT0
*

SrcT0	*'
_output_shapes
:         
Р
?binary_logistic_head/metrics/recall_at_thresholds/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Ё
9binary_logistic_head/metrics/recall_at_thresholds/ReshapeReshape)binary_logistic_head/predictions/logistic?binary_logistic_head/metrics/recall_at_thresholds/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Т
Abinary_logistic_head/metrics/recall_at_thresholds/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
Б
;binary_logistic_head/metrics/recall_at_thresholds/Reshape_1Reshape6binary_logistic_head/metrics/recall_at_thresholds/CastAbinary_logistic_head/metrics/recall_at_thresholds/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
░
7binary_logistic_head/metrics/recall_at_thresholds/ShapeShape9binary_logistic_head/metrics/recall_at_thresholds/Reshape*
out_type0*
T0*
_output_shapes
:
П
Ebinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
С
Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
С
Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
є
?binary_logistic_head/metrics/recall_at_thresholds/strided_sliceStridedSlice7binary_logistic_head/metrics/recall_at_thresholds/ShapeEbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stackGbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_1Gbinary_logistic_head/metrics/recall_at_thresholds/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Д
7binary_logistic_head/metrics/recall_at_thresholds/ConstConst*
dtype0*
valueB*   ?*
_output_shapes
:
К
@binary_logistic_head/metrics/recall_at_thresholds/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
·
<binary_logistic_head/metrics/recall_at_thresholds/ExpandDims
ExpandDims7binary_logistic_head/metrics/recall_at_thresholds/Const@binary_logistic_head/metrics/recall_at_thresholds/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
{
9binary_logistic_head/metrics/recall_at_thresholds/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
ї
7binary_logistic_head/metrics/recall_at_thresholds/stackPack9binary_logistic_head/metrics/recall_at_thresholds/stack/0?binary_logistic_head/metrics/recall_at_thresholds/strided_slice*
_output_shapes
:*

axis *
T0*
N
∙
6binary_logistic_head/metrics/recall_at_thresholds/TileTile<binary_logistic_head/metrics/recall_at_thresholds/ExpandDims7binary_logistic_head/metrics/recall_at_thresholds/stack*

Tmultiples0*
T0*'
_output_shapes
:         
д
@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankRank9binary_logistic_head/metrics/recall_at_thresholds/Reshape*
T0*
_output_shapes
: 
Г
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
ь
?binary_logistic_head/metrics/recall_at_thresholds/transpose/subSub@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankAbinary_logistic_head/metrics/recall_at_thresholds/transpose/sub/y*
T0*
_output_shapes
: 
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╞
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/RangeRangeGbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/start@binary_logistic_head/metrics/recall_at_thresholds/transpose/RankGbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range/delta*

Tidx0*
_output_shapes
:
ё
Abinary_logistic_head/metrics/recall_at_thresholds/transpose/sub_1Sub?binary_logistic_head/metrics/recall_at_thresholds/transpose/subAbinary_logistic_head/metrics/recall_at_thresholds/transpose/Range*
T0*
_output_shapes
:
Е
;binary_logistic_head/metrics/recall_at_thresholds/transpose	Transpose9binary_logistic_head/metrics/recall_at_thresholds/ReshapeAbinary_logistic_head/metrics/recall_at_thresholds/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_1/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
Е
8binary_logistic_head/metrics/recall_at_thresholds/Tile_1Tile;binary_logistic_head/metrics/recall_at_thresholds/transposeBbinary_logistic_head/metrics/recall_at_thresholds/Tile_1/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
ш
9binary_logistic_head/metrics/recall_at_thresholds/GreaterGreater8binary_logistic_head/metrics/recall_at_thresholds/Tile_16binary_logistic_head/metrics/recall_at_thresholds/Tile*
T0*'
_output_shapes
:         
о
<binary_logistic_head/metrics/recall_at_thresholds/LogicalNot
LogicalNot9binary_logistic_head/metrics/recall_at_thresholds/Greater*'
_output_shapes
:         
У
Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_2/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
Е
8binary_logistic_head/metrics/recall_at_thresholds/Tile_2Tile;binary_logistic_head/metrics/recall_at_thresholds/Reshape_1Bbinary_logistic_head/metrics/recall_at_thresholds/Tile_2/multiples*

Tmultiples0*
T0
*'
_output_shapes
:         
Д
7binary_logistic_head/metrics/recall_at_thresholds/zerosConst*
dtype0*
valueB*    *
_output_shapes
:
м
@binary_logistic_head/metrics/recall_at_thresholds/true_positives
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
я
Gbinary_logistic_head/metrics/recall_at_thresholds/true_positives/AssignAssign@binary_logistic_head/metrics/recall_at_thresholds/true_positives7binary_logistic_head/metrics/recall_at_thresholds/zeros*
validate_shape(*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
use_locking(*
T0*
_output_shapes
:
Н
Ebinary_logistic_head/metrics/recall_at_thresholds/true_positives/readIdentity@binary_logistic_head/metrics/recall_at_thresholds/true_positives*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
T0*
_output_shapes
:
ш
<binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd
LogicalAnd8binary_logistic_head/metrics/recall_at_thresholds/Tile_29binary_logistic_head/metrics/recall_at_thresholds/Greater*'
_output_shapes
:         
┬
;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_1Cast<binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd*

DstT0*

SrcT0
*'
_output_shapes
:         
Й
Gbinary_logistic_head/metrics/recall_at_thresholds/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Д
5binary_logistic_head/metrics/recall_at_thresholds/SumSum;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_1Gbinary_logistic_head/metrics/recall_at_thresholds/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
╬
;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd	AssignAdd@binary_logistic_head/metrics/recall_at_thresholds/true_positives5binary_logistic_head/metrics/recall_at_thresholds/Sum*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
use_locking( *
T0*
_output_shapes
:
Ж
9binary_logistic_head/metrics/recall_at_thresholds/zeros_1Const*
dtype0*
valueB*    *
_output_shapes
:
н
Abinary_logistic_head/metrics/recall_at_thresholds/false_negatives
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
Ї
Hbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/AssignAssignAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives9binary_logistic_head/metrics/recall_at_thresholds/zeros_1*
validate_shape(*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
use_locking(*
T0*
_output_shapes
:
Р
Fbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/readIdentityAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
T0*
_output_shapes
:
э
>binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd_1
LogicalAnd8binary_logistic_head/metrics/recall_at_thresholds/Tile_2<binary_logistic_head/metrics/recall_at_thresholds/LogicalNot*'
_output_shapes
:         
─
;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_2Cast>binary_logistic_head/metrics/recall_at_thresholds/LogicalAnd_1*

DstT0*

SrcT0
*'
_output_shapes
:         
Л
Ibinary_logistic_head/metrics/recall_at_thresholds/Sum_1/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
И
7binary_logistic_head/metrics/recall_at_thresholds/Sum_1Sum;binary_logistic_head/metrics/recall_at_thresholds/ToFloat_2Ibinary_logistic_head/metrics/recall_at_thresholds/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
╘
=binary_logistic_head/metrics/recall_at_thresholds/AssignAdd_1	AssignAddAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives7binary_logistic_head/metrics/recall_at_thresholds/Sum_1*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
use_locking( *
T0*
_output_shapes
:
|
7binary_logistic_head/metrics/recall_at_thresholds/add/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
с
5binary_logistic_head/metrics/recall_at_thresholds/addAdd7binary_logistic_head/metrics/recall_at_thresholds/add/xEbinary_logistic_head/metrics/recall_at_thresholds/true_positives/read*
T0*
_output_shapes
:
т
7binary_logistic_head/metrics/recall_at_thresholds/add_1Add5binary_logistic_head/metrics/recall_at_thresholds/addFbinary_logistic_head/metrics/recall_at_thresholds/false_negatives/read*
T0*
_output_shapes
:
ю
>binary_logistic_head/metrics/recall_at_thresholds/recall_valueRealDivEbinary_logistic_head/metrics/recall_at_thresholds/true_positives/read7binary_logistic_head/metrics/recall_at_thresholds/add_1*
T0*
_output_shapes
:
~
9binary_logistic_head/metrics/recall_at_thresholds/add_2/xConst*
dtype0*
valueB
 *Х┐╓3*
_output_shapes
: 
█
7binary_logistic_head/metrics/recall_at_thresholds/add_2Add9binary_logistic_head/metrics/recall_at_thresholds/add_2/x;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd*
T0*
_output_shapes
:
█
7binary_logistic_head/metrics/recall_at_thresholds/add_3Add7binary_logistic_head/metrics/recall_at_thresholds/add_2=binary_logistic_head/metrics/recall_at_thresholds/AssignAdd_1*
T0*
_output_shapes
:
ш
Bbinary_logistic_head/metrics/recall_at_thresholds/recall_update_opRealDiv;binary_logistic_head/metrics/recall_at_thresholds/AssignAdd7binary_logistic_head/metrics/recall_at_thresholds/add_3*
T0*
_output_shapes
:
ж
&binary_logistic_head/metrics/Squeeze_2Squeeze>binary_logistic_head/metrics/recall_at_thresholds/recall_value*
squeeze_dims
 *
T0*
_output_shapes
: 
к
&binary_logistic_head/metrics/Squeeze_3SqueezeBbinary_logistic_head/metrics/recall_at_thresholds/recall_update_op*
squeeze_dims
 *
T0*
_output_shapes
: 
м
>binary_logistic_head/_classification_output_alternatives/ShapeShape.binary_logistic_head/predictions/probabilities*
out_type0*
T0*
_output_shapes
:
Ц
Lbinary_logistic_head/_classification_output_alternatives/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ц
Fbinary_logistic_head/_classification_output_alternatives/strided_sliceStridedSlice>binary_logistic_head/_classification_output_alternatives/ShapeLbinary_logistic_head/_classification_output_alternatives/strided_slice/stackNbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_1Nbinary_logistic_head/_classification_output_alternatives/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
о
@binary_logistic_head/_classification_output_alternatives/Shape_1Shape.binary_logistic_head/predictions/probabilities*
out_type0*
T0*
_output_shapes
:
Ш
Nbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Ъ
Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
а
Hbinary_logistic_head/_classification_output_alternatives/strided_slice_1StridedSlice@binary_logistic_head/_classification_output_alternatives/Shape_1Nbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stackPbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_1Pbinary_logistic_head/_classification_output_alternatives/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
Ж
Dbinary_logistic_head/_classification_output_alternatives/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Ж
Dbinary_logistic_head/_classification_output_alternatives/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
╬
>binary_logistic_head/_classification_output_alternatives/rangeRangeDbinary_logistic_head/_classification_output_alternatives/range/startHbinary_logistic_head/_classification_output_alternatives/strided_slice_1Dbinary_logistic_head/_classification_output_alternatives/range/delta*

Tidx0*#
_output_shapes
:         
Й
Gbinary_logistic_head/_classification_output_alternatives/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
Ш
Cbinary_logistic_head/_classification_output_alternatives/ExpandDims
ExpandDims>binary_logistic_head/_classification_output_alternatives/rangeGbinary_logistic_head/_classification_output_alternatives/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
Л
Ibinary_logistic_head/_classification_output_alternatives/Tile/multiples/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ь
Gbinary_logistic_head/_classification_output_alternatives/Tile/multiplesPackFbinary_logistic_head/_classification_output_alternatives/strided_sliceIbinary_logistic_head/_classification_output_alternatives/Tile/multiples/1*
_output_shapes
:*

axis *
T0*
N
а
=binary_logistic_head/_classification_output_alternatives/TileTileCbinary_logistic_head/_classification_output_alternatives/ExpandDimsGbinary_logistic_head/_classification_output_alternatives/Tile/multiples*

Tmultiples0*
T0*0
_output_shapes
:                  
л
Gbinary_logistic_head/_classification_output_alternatives/classes_tensorAsString=binary_logistic_head/_classification_output_alternatives/Tile*

scientific( *0
_output_shapes
:                  *
	precision         *
width         *
T0*
shortest( *

fill 
Р
$remove_squeezable_dimensions/SqueezeSqueezehash_table_Lookup*
squeeze_dims

         *
T0	*#
_output_shapes
:         
М
EqualEqual(binary_logistic_head/predictions/classes$remove_squeezable_dimensions/Squeeze*
T0	*#
_output_shapes
:         
S
ToFloatCastEqual*

DstT0*

SrcT0
*#
_output_shapes
:         
S
accuracy/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
accuracy/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
м
accuracy/total/AssignAssignaccuracy/totalaccuracy/zeros*
validate_shape(*!
_class
loc:@accuracy/total*
use_locking(*
T0*
_output_shapes
: 
s
accuracy/total/readIdentityaccuracy/total*!
_class
loc:@accuracy/total*
T0*
_output_shapes
: 
U
accuracy/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
r
accuracy/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
о
accuracy/count/AssignAssignaccuracy/countaccuracy/zeros_1*
validate_shape(*!
_class
loc:@accuracy/count*
use_locking(*
T0*
_output_shapes
: 
s
accuracy/count/readIdentityaccuracy/count*!
_class
loc:@accuracy/count*
T0*
_output_shapes
: 
O
accuracy/SizeSizeToFloat*
out_type0*
T0*
_output_shapes
: 
Y
accuracy/ToFloat_1Castaccuracy/Size*

DstT0*

SrcT0*
_output_shapes
: 
X
accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
j
accuracy/SumSumToFloataccuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ф
accuracy/AssignAdd	AssignAddaccuracy/totalaccuracy/Sum*!
_class
loc:@accuracy/total*
use_locking( *
T0*
_output_shapes
: 
ж
accuracy/AssignAdd_1	AssignAddaccuracy/countaccuracy/ToFloat_1^ToFloat*!
_class
loc:@accuracy/count*
use_locking( *
T0*
_output_shapes
: 
W
accuracy/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
e
accuracy/GreaterGreateraccuracy/count/readaccuracy/Greater/y*
T0*
_output_shapes
: 
f
accuracy/truedivRealDivaccuracy/total/readaccuracy/count/read*
T0*
_output_shapes
: 
U
accuracy/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
o
accuracy/valueSelectaccuracy/Greateraccuracy/truedivaccuracy/value/e*
T0*
_output_shapes
: 
Y
accuracy/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
j
accuracy/Greater_1Greateraccuracy/AssignAdd_1accuracy/Greater_1/y*
T0*
_output_shapes
: 
h
accuracy/truediv_1RealDivaccuracy/AssignAddaccuracy/AssignAdd_1*
T0*
_output_shapes
: 
Y
accuracy/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
{
accuracy/update_opSelectaccuracy/Greater_1accuracy/truediv_1accuracy/update_op/e*
T0*
_output_shapes
: 
╟

group_depsNoOp.^binary_logistic_head/metrics/mean_3/update_op,^binary_logistic_head/metrics/mean/update_op+^binary_logistic_head/metrics/auc/update_op2^binary_logistic_head/metrics/accuracy_1/update_op'^binary_logistic_head/metrics/Squeeze_3.^binary_logistic_head/metrics/mean_1/update_op^accuracy/update_op-^binary_logistic_head/metrics/auc_1/update_op'^binary_logistic_head/metrics/Squeeze_1.^binary_logistic_head/metrics/mean_2/update_op
{
eval_step/Initializer/zerosConst*
dtype0	*
_class
loc:@eval_step*
value	B	 R *
_output_shapes
: 
Л
	eval_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@eval_step*
shared_name 
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
validate_shape(*
_class
loc:@eval_step*
use_locking(*
T0	*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0	*
_output_shapes
: 
Q
AssignAdd/valueConst*
dtype0	*
value	B	 R*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_class
loc:@eval_step*
use_locking( *
T0	*
_output_shapes
: 
є
initNoOp^global_step/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign0^linear/linear_model/alpha/weights/part_0/Assign/^linear/linear_model/beta/weights/part_0/Assign/^linear/linear_model/bias_weights/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
╦
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
╜
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
█
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized(linear/linear_model/alpha/weights/part_0*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized'linear/linear_model/beta/weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized'linear/linear_model/bias_weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized'binary_logistic_head/metrics/mean/total*
dtype0*:
_class0
.,loc:@binary_logistic_head/metrics/mean/total*
_output_shapes
: 
┌
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized'binary_logistic_head/metrics/mean/count*
dtype0*:
_class0
.,loc:@binary_logistic_head/metrics/mean/count*
_output_shapes
: 
т
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized+binary_logistic_head/metrics/accuracy/total*
dtype0*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/total*
_output_shapes
: 
т
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized+binary_logistic_head/metrics/accuracy/count*
dtype0*>
_class4
20loc:@binary_logistic_head/metrics/accuracy/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized)binary_logistic_head/metrics/mean_1/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized)binary_logistic_head/metrics/mean_1/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_1/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized)binary_logistic_head/metrics/mean_2/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized)binary_logistic_head/metrics/mean_2/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_2/count*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized)binary_logistic_head/metrics/mean_3/total*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/total*
_output_shapes
: 
▐
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized)binary_logistic_head/metrics/mean_3/count*
dtype0*<
_class2
0.loc:@binary_logistic_head/metrics/mean_3/count*
_output_shapes
: 
ъ
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized/binary_logistic_head/metrics/auc/true_positives*
dtype0*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_positives*
_output_shapes
: 
ь
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized0binary_logistic_head/metrics/auc/false_negatives*
dtype0*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_negatives*
_output_shapes
: 
ъ
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized/binary_logistic_head/metrics/auc/true_negatives*
dtype0*B
_class8
64loc:@binary_logistic_head/metrics/auc/true_negatives*
_output_shapes
: 
ь
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized0binary_logistic_head/metrics/auc/false_positives*
dtype0*C
_class9
75loc:@binary_logistic_head/metrics/auc/false_positives*
_output_shapes
: 
ю
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized1binary_logistic_head/metrics/auc_1/true_positives*
dtype0*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_positives*
_output_shapes
: 
Ё
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized2binary_logistic_head/metrics/auc_1/false_negatives*
dtype0*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_negatives*
_output_shapes
: 
ю
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized1binary_logistic_head/metrics/auc_1/true_negatives*
dtype0*D
_class:
86loc:@binary_logistic_head/metrics/auc_1/true_negatives*
_output_shapes
: 
Ё
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized2binary_logistic_head/metrics/auc_1/false_positives*
dtype0*E
_class;
97loc:@binary_logistic_head/metrics/auc_1/false_positives*
_output_shapes
: 
ц
7report_uninitialized_variables/IsVariableInitialized_30IsVariableInitialized-binary_logistic_head/metrics/accuracy_1/total*
dtype0*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/total*
_output_shapes
: 
ц
7report_uninitialized_variables/IsVariableInitialized_31IsVariableInitialized-binary_logistic_head/metrics/accuracy_1/count*
dtype0*@
_class6
42loc:@binary_logistic_head/metrics/accuracy_1/count*
_output_shapes
: 
Т
7report_uninitialized_variables/IsVariableInitialized_32IsVariableInitializedCbinary_logistic_head/metrics/precision_at_thresholds/true_positives*
dtype0*V
_classL
JHloc:@binary_logistic_head/metrics/precision_at_thresholds/true_positives*
_output_shapes
: 
Ф
7report_uninitialized_variables/IsVariableInitialized_33IsVariableInitializedDbinary_logistic_head/metrics/precision_at_thresholds/false_positives*
dtype0*W
_classM
KIloc:@binary_logistic_head/metrics/precision_at_thresholds/false_positives*
_output_shapes
: 
М
7report_uninitialized_variables/IsVariableInitialized_34IsVariableInitialized@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
dtype0*S
_classI
GEloc:@binary_logistic_head/metrics/recall_at_thresholds/true_positives*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_35IsVariableInitializedAbinary_logistic_head/metrics/recall_at_thresholds/false_negatives*
dtype0*T
_classJ
HFloc:@binary_logistic_head/metrics/recall_at_thresholds/false_negatives*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_36IsVariableInitializedaccuracy/total*
dtype0*!
_class
loc:@accuracy/total*
_output_shapes
: 
и
7report_uninitialized_variables/IsVariableInitialized_37IsVariableInitializedaccuracy/count*
dtype0*!
_class
loc:@accuracy/count*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_38IsVariableInitialized	eval_step*
dtype0	*
_class
loc:@eval_step*
_output_shapes
: 
Й
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_297report_uninitialized_variables/IsVariableInitialized_307report_uninitialized_variables/IsVariableInitialized_317report_uninitialized_variables/IsVariableInitialized_327report_uninitialized_variables/IsVariableInitialized_337report_uninitialized_variables/IsVariableInitialized_347report_uninitialized_variables/IsVariableInitialized_357report_uninitialized_variables/IsVariableInitialized_367report_uninitialized_variables/IsVariableInitialized_377report_uninitialized_variables/IsVariableInitialized_38*
_output_shapes
:'*

axis *
T0
*
N'
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:'
╨
$report_uninitialized_variables/ConstConst*
dtype0*ў
valueэBъ'Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B(linear/linear_model/alpha/weights/part_0B'linear/linear_model/beta/weights/part_0B'linear/linear_model/bias_weights/part_0B'binary_logistic_head/metrics/mean/totalB'binary_logistic_head/metrics/mean/countB+binary_logistic_head/metrics/accuracy/totalB+binary_logistic_head/metrics/accuracy/countB)binary_logistic_head/metrics/mean_1/totalB)binary_logistic_head/metrics/mean_1/countB)binary_logistic_head/metrics/mean_2/totalB)binary_logistic_head/metrics/mean_2/countB)binary_logistic_head/metrics/mean_3/totalB)binary_logistic_head/metrics/mean_3/countB/binary_logistic_head/metrics/auc/true_positivesB0binary_logistic_head/metrics/auc/false_negativesB/binary_logistic_head/metrics/auc/true_negativesB0binary_logistic_head/metrics/auc/false_positivesB1binary_logistic_head/metrics/auc_1/true_positivesB2binary_logistic_head/metrics/auc_1/false_negativesB1binary_logistic_head/metrics/auc_1/true_negativesB2binary_logistic_head/metrics/auc_1/false_positivesB-binary_logistic_head/metrics/accuracy_1/totalB-binary_logistic_head/metrics/accuracy_1/countBCbinary_logistic_head/metrics/precision_at_thresholds/true_positivesBDbinary_logistic_head/metrics/precision_at_thresholds/false_positivesB@binary_logistic_head/metrics/recall_at_thresholds/true_positivesBAbinary_logistic_head/metrics/recall_at_thresholds/false_negativesBaccuracy/totalBaccuracy/countB	eval_step*
_output_shapes
:'
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:'*
_output_shapes
:
Й
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┘
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
М
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ї
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:'*
_output_shapes
:
Л
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
с
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
п
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
л
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╦
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:'
О
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
█
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:'
Ъ
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:         
╢
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
В
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:         *

Tidx0*
T0*
N
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
═
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
┐
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
▌
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitialized(linear/linear_model/alpha/weights/part_0*
dtype0*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
_output_shapes
: 
▄
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitialized'linear/linear_model/beta/weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
_output_shapes
: 
▄
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitialized'linear/linear_model/bias_weights/part_0*
dtype0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: 
а
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_11*
_output_shapes
:*

axis *
T0
*
N
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
ў
&report_uninitialized_variables_1/ConstConst*
dtype0*Ь
valueТBПBglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0B(linear/linear_model/alpha/weights/part_0B'linear/linear_model/beta/weights/part_0B'linear/linear_model/bias_weights/part_0*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Л
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
у
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
О
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
√
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
Н
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ы
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
│
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
│
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╤
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
Р
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
с
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
Ю
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:         
║
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:         
И
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:         
┴
init_2NoOp/^binary_logistic_head/metrics/mean/total/Assign/^binary_logistic_head/metrics/mean/count/Assign3^binary_logistic_head/metrics/accuracy/total/Assign3^binary_logistic_head/metrics/accuracy/count/Assign1^binary_logistic_head/metrics/mean_1/total/Assign1^binary_logistic_head/metrics/mean_1/count/Assign1^binary_logistic_head/metrics/mean_2/total/Assign1^binary_logistic_head/metrics/mean_2/count/Assign1^binary_logistic_head/metrics/mean_3/total/Assign1^binary_logistic_head/metrics/mean_3/count/Assign7^binary_logistic_head/metrics/auc/true_positives/Assign8^binary_logistic_head/metrics/auc/false_negatives/Assign7^binary_logistic_head/metrics/auc/true_negatives/Assign8^binary_logistic_head/metrics/auc/false_positives/Assign9^binary_logistic_head/metrics/auc_1/true_positives/Assign:^binary_logistic_head/metrics/auc_1/false_negatives/Assign9^binary_logistic_head/metrics/auc_1/true_negatives/Assign:^binary_logistic_head/metrics/auc_1/false_positives/Assign5^binary_logistic_head/metrics/accuracy_1/total/Assign5^binary_logistic_head/metrics/accuracy_1/count/AssignK^binary_logistic_head/metrics/precision_at_thresholds/true_positives/AssignL^binary_logistic_head/metrics/precision_at_thresholds/false_positives/AssignH^binary_logistic_head/metrics/recall_at_thresholds/true_positives/AssignI^binary_logistic_head/metrics/recall_at_thresholds/false_negatives/Assign^accuracy/total/Assign^accuracy/count/Assign^eval_step/Assign
∙
init_all_tablesNoOp&^string_to_index/hash_table/table_init^^dnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/table_init\^dnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/table_init=^linear/linear_model/alpha/alpha_lookup/hash_table/table_init;^linear/linear_model/beta/beta_lookup/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
Я
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_5000_full-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation-dnn/dnn/hiddenlayer_2/fraction_of_zero_values dnn/dnn/hiddenlayer_2/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation%linear/linear/fraction_of_zero_valueslinear/linear/activation"binary_logistic_head/ScalarSummary*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_5a457b3906f24157867a6202b2d6f0c4/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ь
save/SaveV2/tensor_namesConst*
dtype0*╧
value┼B┬Bdnn/hiddenlayer_0/biasesBdnn/hiddenlayer_0/weightsBdnn/hiddenlayer_1/biasesBdnn/hiddenlayer_1/weightsBdnn/hiddenlayer_2/biasesBdnn/hiddenlayer_2/weightsBdnn/logits/biasesBdnn/logits/weightsBglobal_stepB!linear/linear_model/alpha/weightsB linear/linear_model/beta/weightsB linear/linear_model/bias_weights*
_output_shapes
:
ё
save/SaveV2/shape_and_slicesConst*
dtype0*а
valueЦBУB	128 0,128B6 128 0,6:0,128B64 0,64B128 64 0,128:0,64B32 0,32B64 32 0,64:0,32B1 0,1B32 1 0,32:0,1B B2 1 0,2:0,1B2 1 0,2:0,1B1 0,1*
_output_shapes
:
╜
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read%dnn/hiddenlayer_0/weights/part_0/read$dnn/hiddenlayer_1/biases/part_0/read%dnn/hiddenlayer_1/weights/part_0/read$dnn/hiddenlayer_2/biases/part_0/read%dnn/hiddenlayer_2/weights/part_0/readdnn/logits/biases/part_0/readdnn/logits/weights/part_0/readglobal_step-linear/linear_model/alpha/weights/part_0/read,linear/linear_model/beta/weights/part_0/read,linear/linear_model/bias_weights/part_0/read*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
T0*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
|
save/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
q
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB	128 0,128*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
╔
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes	
:А

save/RestoreV2_1/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*$
valueBB6 128 0,6:0,128*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_1Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_1*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А
~
save/RestoreV2_2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueBB64 0,64*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_2Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:@

save/RestoreV2_3/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
{
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*&
valueBB128 64 0,128:0,64*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
╙
save/Assign_3Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes
:	А@
~
save/RestoreV2_4/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_2/biases*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueBB32 0,32*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
╠
save/Assign_4Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
: 

save/RestoreV2_5/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights*
_output_shapes
:
y
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*$
valueBB64 32 0,64:0,32*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
╥
save/Assign_5Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_5*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:@ 
w
save/RestoreV2_6/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
o
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
╛
save/Assign_6Assigndnn/logits/biases/part_0save/RestoreV2_6*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
x
save/RestoreV2_7/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
w
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*"
valueBB32 1 0,32:0,1*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
─
save/Assign_7Assigndnn/logits/weights/part_0save/RestoreV2_7*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

: 
q
save/RestoreV2_8/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2	*
_output_shapes
:
а
save/Assign_8Assignglobal_stepsave/RestoreV2_8*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
З
save/RestoreV2_9/tensor_namesConst*
dtype0*6
value-B+B!linear/linear_model/alpha/weights*
_output_shapes
:
u
!save/RestoreV2_9/shape_and_slicesConst*
dtype0* 
valueBB2 1 0,2:0,1*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
т
save/Assign_9Assign(linear/linear_model/alpha/weights/part_0save/RestoreV2_9*
validate_shape(*;
_class1
/-loc:@linear/linear_model/alpha/weights/part_0*
use_locking(*
T0*
_output_shapes

:
З
save/RestoreV2_10/tensor_namesConst*
dtype0*5
value,B*B linear/linear_model/beta/weights*
_output_shapes
:
v
"save/RestoreV2_10/shape_and_slicesConst*
dtype0* 
valueBB2 1 0,2:0,1*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
т
save/Assign_10Assign'linear/linear_model/beta/weights/part_0save/RestoreV2_10*
validate_shape(*:
_class0
.,loc:@linear/linear_model/beta/weights/part_0*
use_locking(*
T0*
_output_shapes

:
З
save/RestoreV2_11/tensor_namesConst*
dtype0*5
value,B*B linear/linear_model/bias_weights*
_output_shapes
:
p
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
▐
save/Assign_11Assign'linear/linear_model/bias_weights/part_0save/RestoreV2_11*
validate_shape(*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
use_locking(*
T0*
_output_shapes
:
┌
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11
-
save/restore_allNoOp^save/restore_shard""U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0" 
global_step

global_step:0"Ч
trainable_variables №
б
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0")
dnn/hiddenlayer_0/weightsА  "А
Ъ
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"%
dnn/hiddenlayer_0/biasesА "А
б
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0")
dnn/hiddenlayer_1/weightsА@  "А@
Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases@ "@
Я
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights@   "@ 
Ш
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases  " 
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights   " 
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases "
┐
*linear/linear_model/alpha/weights/part_0:0/linear/linear_model/alpha/weights/part_0/Assign/linear/linear_model/alpha/weights/part_0/read:0"/
!linear/linear_model/alpha/weights  "
╗
)linear/linear_model/beta/weights/part_0:0.linear/linear_model/beta/weights/part_0/Assign.linear/linear_model/beta/weights/part_0/read:0".
 linear/linear_model/beta/weights  "
╕
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign.linear/linear_model/bias_weights/part_0/read:0"+
 linear/linear_model/bias_weights ""!
local_init_op

group_deps_2"╞
	variables╕╡
7
global_step:0global_step/Assignglobal_step/read:0
б
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0")
dnn/hiddenlayer_0/weightsА  "А
Ъ
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"%
dnn/hiddenlayer_0/biasesА "А
б
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0")
dnn/hiddenlayer_1/weightsА@  "А@
Ш
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biases@ "@
Я
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weights@   "@ 
Ш
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases  " 
Г
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights   " 
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases "
┐
*linear/linear_model/alpha/weights/part_0:0/linear/linear_model/alpha/weights/part_0/Assign/linear/linear_model/alpha/weights/part_0/read:0"/
!linear/linear_model/alpha/weights  "
╗
)linear/linear_model/beta/weights/part_0:0.linear/linear_model/beta/weights/part_0/Assign.linear/linear_model/beta/weights/part_0/read:0".
 linear/linear_model/beta/weights  "
╕
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign.linear/linear_model/bias_weights/part_0/read:0"+
 linear/linear_model/bias_weights ""Щ
dnnС
О
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"З
	summaries∙
Ў
$input_producer/fraction_of_32_full:0
batch/fraction_of_5000_full:0
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
'linear/linear/fraction_of_zero_values:0
linear/linear/activation:0
$binary_logistic_head/ScalarSummary:0"и

local_variablesФ

С

)binary_logistic_head/metrics/mean/total:0
)binary_logistic_head/metrics/mean/count:0
-binary_logistic_head/metrics/accuracy/total:0
-binary_logistic_head/metrics/accuracy/count:0
+binary_logistic_head/metrics/mean_1/total:0
+binary_logistic_head/metrics/mean_1/count:0
+binary_logistic_head/metrics/mean_2/total:0
+binary_logistic_head/metrics/mean_2/count:0
+binary_logistic_head/metrics/mean_3/total:0
+binary_logistic_head/metrics/mean_3/count:0
1binary_logistic_head/metrics/auc/true_positives:0
2binary_logistic_head/metrics/auc/false_negatives:0
1binary_logistic_head/metrics/auc/true_negatives:0
2binary_logistic_head/metrics/auc/false_positives:0
3binary_logistic_head/metrics/auc_1/true_positives:0
4binary_logistic_head/metrics/auc_1/false_negatives:0
3binary_logistic_head/metrics/auc_1/true_negatives:0
4binary_logistic_head/metrics/auc_1/false_positives:0
/binary_logistic_head/metrics/accuracy_1/total:0
/binary_logistic_head/metrics/accuracy_1/count:0
Ebinary_logistic_head/metrics/precision_at_thresholds/true_positives:0
Fbinary_logistic_head/metrics/precision_at_thresholds/false_positives:0
Bbinary_logistic_head/metrics/recall_at_thresholds/true_positives:0
Cbinary_logistic_head/metrics/recall_at_thresholds/false_negatives:0
accuracy/total:0
accuracy/count:0
eval_step:0"Ў
table_initializerр
▌
%string_to_index/hash_table/table_init
]dnn/input_from_feature_columns/input_layer/alpha_indicator/alpha_lookup/hash_table/table_init
[dnn/input_from_feature_columns/input_layer/beta_indicator/beta_lookup/hash_table/table_init
<linear/linear_model/alpha/alpha_lookup/hash_table/table_init
:linear/linear_model/beta/beta_lookup/hash_table/table_init"ф
queue_runners╥╧
К
input_producer)input_producer/input_producer_EnqueueMany#input_producer/input_producer_Close"%input_producer/input_producer_Close_1*
┐
batch/fifo_queuebatch/fifo_queue_EnqueueManybatch/fifo_queue_EnqueueManybatch/fifo_queue_EnqueueManybatch/fifo_queue_EnqueueManybatch/fifo_queue_Close"batch/fifo_queue_Close_1*"Р
linearЕ
В
*linear/linear_model/alpha/weights/part_0:0
)linear/linear_model/beta/weights/part_0:0
)linear/linear_model/bias_weights/part_0:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"&

summary_op

Merge/MergeSummary:0"
	eval_step

eval_step:0"
ready_op


concat:0"з
model_variablesУ
Р
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0
*linear/linear_model/alpha/weights/part_0:0
)linear/linear_model/beta/weights/part_0:0
)linear/linear_model/bias_weights/part_0:0"
init_op

group_deps_1((иKG      ╝єjC	╥М╦< l╓AЁ*╕
#
accuracy/baseline_label_mean ?

lossгb0?


auc є?
'
 accuracy/threshold_0.500000_mean№o?
.
'recall/positive_threshold_0.500000_mean.n
?

labels/prediction_meanД >

accuracy№o?

auc_precision_recall\м?
1
*precision/positive_threshold_0.500000_meanLH?

labels/actual_label_mean ?iO╤┌