??'
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-0-gc256c071bb28֝&
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*%
shared_nameembedding/embeddings
?
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*!
_output_shapes
:???*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_namelstm/lstm_cell_8/kernel
?
+lstm/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell_8/kernel* 
_output_shapes
:
??*
dtype0
?
!lstm/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*2
shared_name#!lstm/lstm_cell_8/recurrent_kernel
?
5lstm/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_namelstm/lstm_cell_8/bias
|
)lstm/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell_8/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*!
_output_shapes
:???*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/lstm/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/lstm/lstm_cell_8/kernel/m
?
2Adam/lstm/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_8/kernel/m* 
_output_shapes
:
??*
dtype0
?
(Adam/lstm/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/lstm/lstm_cell_8/recurrent_kernel/m
?
<Adam/lstm/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/lstm/lstm_cell_8/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/lstm/lstm_cell_8/bias/m
?
0Adam/lstm/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_8/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*!
_output_shapes
:???*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/lstm/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name Adam/lstm/lstm_cell_8/kernel/v
?
2Adam/lstm/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_8/kernel/v* 
_output_shapes
:
??*
dtype0
?
(Adam/lstm/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(Adam/lstm/lstm_cell_8/recurrent_kernel/v
?
<Adam/lstm/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/lstm/lstm_cell_8/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nameAdam/lstm/lstm_cell_8/bias/v
?
0Adam/lstm/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell_8/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?)B?) B?)
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratemWmXmY%mZ&m['m\v]v^v_%v`&va'vb
*
0
%1
&2
'3
4
5
*
0
%1
&2
'3
4
5
 
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
?
7
state_size

%kernel
&recurrent_kernel
'bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
 

%0
&1
'2

%0
&1
'2
 
?

<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm/lstm_cell_8/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!lstm/lstm_cell_8/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell_8/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

G0
H1
 
 
 
 
 
 
 
 
 
 
 
 
 

%0
&1
'2

%0
&1
'2
 
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 

0
 
 
 
 
 
 
 
 
4
	Ntotal
	Ocount
P	variables
Q	keras_api
D
	Rtotal
	Scount
T
_fn_kwargs
U	variables
V	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

P	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

U	variables
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm/lstm_cell_8/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/lstm/lstm_cell_8/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell_8/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm/lstm_cell_8/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/lstm/lstm_cell_8/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell_8/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_embedding_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingslstm/lstm_cell_8/kernellstm/lstm_cell_8/bias!lstm/lstm_cell_8/recurrent_kerneldense/kernel
dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_75042
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+lstm/lstm_cell_8/kernel/Read/ReadVariableOp5lstm/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp)lstm/lstm_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp2Adam/lstm/lstm_cell_8/kernel/m/Read/ReadVariableOp<Adam/lstm/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp0Adam/lstm/lstm_cell_8/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp2Adam/lstm/lstm_cell_8/kernel/v/Read/ReadVariableOp<Adam/lstm/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp0Adam/lstm/lstm_cell_8/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_77820
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell_8/kernel!lstm/lstm_cell_8/recurrent_kernellstm/lstm_cell_8/biastotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/lstm_cell_8/kernel/m(Adam/lstm/lstm_cell_8/recurrent_kernel/mAdam/lstm/lstm_cell_8/bias/mAdam/embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/lstm_cell_8/kernel/v(Adam/lstm/lstm_cell_8/recurrent_kernel/vAdam/lstm/lstm_cell_8/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_77911??%
??
?
 __inference__wrapped_model_73446
embedding_input@
+sequential_embedding_embedding_lookup_73144:???M
9sequential_lstm_lstm_cell_8_split_readvariableop_resource:
??J
;sequential_lstm_lstm_cell_8_split_1_readvariableop_resource:	?G
3sequential_lstm_lstm_cell_8_readvariableop_resource:
??B
/sequential_dense_matmul_readvariableop_resource:	?>
0sequential_dense_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?*sequential/lstm/lstm_cell_8/ReadVariableOp?,sequential/lstm/lstm_cell_8/ReadVariableOp_1?,sequential/lstm/lstm_cell_8/ReadVariableOp_2?,sequential/lstm/lstm_cell_8/ReadVariableOp_3?0sequential/lstm/lstm_cell_8/split/ReadVariableOp?2sequential/lstm/lstm_cell_8/split_1/ReadVariableOp?sequential/lstm/whilet
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*(
_output_shapes
:???????????
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_73144sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/73144*-
_output_shapes
:???????????*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/73144*-
_output_shapes
:????????????
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
%sequential/spatial_dropout1d/IdentityIdentity9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:???????????s
sequential/lstm/ShapeShape.sequential/spatial_dropout1d/Identity:output:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: a
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????`
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: a
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: c
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????s
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose	Transpose.sequential/spatial_dropout1d/Identity:output:0'sequential/lstm/transpose/perm:output:0*
T0*-
_output_shapes
:???????????d
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:o
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???o
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
+sequential/lstm/lstm_cell_8/ones_like/ShapeShape(sequential/lstm/strided_slice_2:output:0*
T0*
_output_shapes
:p
+sequential/lstm/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
%sequential/lstm/lstm_cell_8/ones_likeFill4sequential/lstm/lstm_cell_8/ones_like/Shape:output:04sequential/lstm/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????{
-sequential/lstm/lstm_cell_8/ones_like_1/ShapeShapesequential/lstm/zeros:output:0*
T0*
_output_shapes
:r
-sequential/lstm/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'sequential/lstm/lstm_cell_8/ones_like_1Fill6sequential/lstm/lstm_cell_8/ones_like_1/Shape:output:06sequential/lstm/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell_8/mulMul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_1Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_2Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_3Mul(sequential/lstm/strided_slice_2:output:0.sequential/lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????m
+sequential/lstm/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
0sequential/lstm/lstm_cell_8/split/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
!sequential/lstm/lstm_cell_8/splitSplit4sequential/lstm/lstm_cell_8/split/split_dim:output:08sequential/lstm/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
"sequential/lstm/lstm_cell_8/MatMulMatMul#sequential/lstm/lstm_cell_8/mul:z:0*sequential/lstm/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
$sequential/lstm/lstm_cell_8/MatMul_1MatMul%sequential/lstm/lstm_cell_8/mul_1:z:0*sequential/lstm/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
$sequential/lstm/lstm_cell_8/MatMul_2MatMul%sequential/lstm/lstm_cell_8/mul_2:z:0*sequential/lstm/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
$sequential/lstm/lstm_cell_8/MatMul_3MatMul%sequential/lstm/lstm_cell_8/mul_3:z:0*sequential/lstm/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????o
-sequential/lstm/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
2sequential/lstm/lstm_cell_8/split_1/ReadVariableOpReadVariableOp;sequential_lstm_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#sequential/lstm/lstm_cell_8/split_1Split6sequential/lstm/lstm_cell_8/split_1/split_dim:output:0:sequential/lstm/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
#sequential/lstm/lstm_cell_8/BiasAddBiasAdd,sequential/lstm/lstm_cell_8/MatMul:product:0,sequential/lstm/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
%sequential/lstm/lstm_cell_8/BiasAdd_1BiasAdd.sequential/lstm/lstm_cell_8/MatMul_1:product:0,sequential/lstm/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
%sequential/lstm/lstm_cell_8/BiasAdd_2BiasAdd.sequential/lstm/lstm_cell_8/MatMul_2:product:0,sequential/lstm/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
%sequential/lstm/lstm_cell_8/BiasAdd_3BiasAdd.sequential/lstm/lstm_cell_8/MatMul_3:product:0,sequential/lstm/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_4Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_5Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_6Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/mul_7Mulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
*sequential/lstm/lstm_cell_8/ReadVariableOpReadVariableOp3sequential_lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
/sequential/lstm/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
1sequential/lstm/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
1sequential/lstm/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
)sequential/lstm/lstm_cell_8/strided_sliceStridedSlice2sequential/lstm/lstm_cell_8/ReadVariableOp:value:08sequential/lstm/lstm_cell_8/strided_slice/stack:output:0:sequential/lstm/lstm_cell_8/strided_slice/stack_1:output:0:sequential/lstm/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
$sequential/lstm/lstm_cell_8/MatMul_4MatMul%sequential/lstm/lstm_cell_8/mul_4:z:02sequential/lstm/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
sequential/lstm/lstm_cell_8/addAddV2,sequential/lstm/lstm_cell_8/BiasAdd:output:0.sequential/lstm/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????f
!sequential/lstm/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>h
#sequential/lstm/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
!sequential/lstm/lstm_cell_8/Mul_8Mul#sequential/lstm/lstm_cell_8/add:z:0*sequential/lstm/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/Add_1AddV2%sequential/lstm/lstm_cell_8/Mul_8:z:0,sequential/lstm/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????x
3sequential/lstm/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1sequential/lstm/lstm_cell_8/clip_by_value/MinimumMinimum%sequential/lstm/lstm_cell_8/Add_1:z:0<sequential/lstm/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????p
+sequential/lstm/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
)sequential/lstm/lstm_cell_8/clip_by_valueMaximum5sequential/lstm/lstm_cell_8/clip_by_value/Minimum:z:04sequential/lstm/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
,sequential/lstm/lstm_cell_8/ReadVariableOp_1ReadVariableOp3sequential_lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
1sequential/lstm/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
3sequential/lstm/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
3sequential/lstm/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
+sequential/lstm/lstm_cell_8/strided_slice_1StridedSlice4sequential/lstm/lstm_cell_8/ReadVariableOp_1:value:0:sequential/lstm/lstm_cell_8/strided_slice_1/stack:output:0<sequential/lstm/lstm_cell_8/strided_slice_1/stack_1:output:0<sequential/lstm/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
$sequential/lstm/lstm_cell_8/MatMul_5MatMul%sequential/lstm/lstm_cell_8/mul_5:z:04sequential/lstm/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/add_2AddV2.sequential/lstm/lstm_cell_8/BiasAdd_1:output:0.sequential/lstm/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????h
#sequential/lstm/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>h
#sequential/lstm/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
!sequential/lstm/lstm_cell_8/Mul_9Mul%sequential/lstm/lstm_cell_8/add_2:z:0,sequential/lstm/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/Add_3AddV2%sequential/lstm/lstm_cell_8/Mul_9:z:0,sequential/lstm/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????z
5sequential/lstm/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
3sequential/lstm/lstm_cell_8/clip_by_value_1/MinimumMinimum%sequential/lstm/lstm_cell_8/Add_3:z:0>sequential/lstm/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????r
-sequential/lstm/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+sequential/lstm/lstm_cell_8/clip_by_value_1Maximum7sequential/lstm/lstm_cell_8/clip_by_value_1/Minimum:z:06sequential/lstm/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
"sequential/lstm/lstm_cell_8/mul_10Mul/sequential/lstm/lstm_cell_8/clip_by_value_1:z:0 sequential/lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
,sequential/lstm/lstm_cell_8/ReadVariableOp_2ReadVariableOp3sequential_lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
1sequential/lstm/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
3sequential/lstm/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
3sequential/lstm/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
+sequential/lstm/lstm_cell_8/strided_slice_2StridedSlice4sequential/lstm/lstm_cell_8/ReadVariableOp_2:value:0:sequential/lstm/lstm_cell_8/strided_slice_2/stack:output:0<sequential/lstm/lstm_cell_8/strided_slice_2/stack_1:output:0<sequential/lstm/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
$sequential/lstm/lstm_cell_8/MatMul_6MatMul%sequential/lstm/lstm_cell_8/mul_6:z:04sequential/lstm/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/add_4AddV2.sequential/lstm/lstm_cell_8/BiasAdd_2:output:0.sequential/lstm/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
 sequential/lstm/lstm_cell_8/TanhTanh%sequential/lstm/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
"sequential/lstm/lstm_cell_8/mul_11Mul-sequential/lstm/lstm_cell_8/clip_by_value:z:0$sequential/lstm/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/add_5AddV2&sequential/lstm/lstm_cell_8/mul_10:z:0&sequential/lstm/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
,sequential/lstm/lstm_cell_8/ReadVariableOp_3ReadVariableOp3sequential_lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
1sequential/lstm/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
3sequential/lstm/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
3sequential/lstm/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
+sequential/lstm/lstm_cell_8/strided_slice_3StridedSlice4sequential/lstm/lstm_cell_8/ReadVariableOp_3:value:0:sequential/lstm/lstm_cell_8/strided_slice_3/stack:output:0<sequential/lstm/lstm_cell_8/strided_slice_3/stack_1:output:0<sequential/lstm/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
$sequential/lstm/lstm_cell_8/MatMul_7MatMul%sequential/lstm/lstm_cell_8/mul_7:z:04sequential/lstm/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/add_6AddV2.sequential/lstm/lstm_cell_8/BiasAdd_3:output:0.sequential/lstm/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????h
#sequential/lstm/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>h
#sequential/lstm/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
"sequential/lstm/lstm_cell_8/Mul_12Mul%sequential/lstm/lstm_cell_8/add_6:z:0,sequential/lstm/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
!sequential/lstm/lstm_cell_8/Add_7AddV2&sequential/lstm/lstm_cell_8/Mul_12:z:0,sequential/lstm/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????z
5sequential/lstm/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
3sequential/lstm/lstm_cell_8/clip_by_value_2/MinimumMinimum%sequential/lstm/lstm_cell_8/Add_7:z:0>sequential/lstm/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????r
-sequential/lstm/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
+sequential/lstm/lstm_cell_8/clip_by_value_2Maximum7sequential/lstm/lstm_cell_8/clip_by_value_2/Minimum:z:06sequential/lstm/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
"sequential/lstm/lstm_cell_8/Tanh_1Tanh%sequential/lstm/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
"sequential/lstm/lstm_cell_8/mul_13Mul/sequential/lstm/lstm_cell_8/clip_by_value_2:z:0&sequential/lstm/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???V
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????d
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_lstm_lstm_cell_8_split_readvariableop_resource;sequential_lstm_lstm_cell_8_split_1_readvariableop_resource3sequential_lstm_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 sequential_lstm_while_body_73285*,
cond$R"
 sequential_lstm_while_cond_73284*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0x
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????q
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:????????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential/dense/MatMulMatMul(sequential/lstm/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup+^sequential/lstm/lstm_cell_8/ReadVariableOp-^sequential/lstm/lstm_cell_8/ReadVariableOp_1-^sequential/lstm/lstm_cell_8/ReadVariableOp_2-^sequential/lstm/lstm_cell_8/ReadVariableOp_31^sequential/lstm/lstm_cell_8/split/ReadVariableOp3^sequential/lstm/lstm_cell_8/split_1/ReadVariableOp^sequential/lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2X
*sequential/lstm/lstm_cell_8/ReadVariableOp*sequential/lstm/lstm_cell_8/ReadVariableOp2\
,sequential/lstm/lstm_cell_8/ReadVariableOp_1,sequential/lstm/lstm_cell_8/ReadVariableOp_12\
,sequential/lstm/lstm_cell_8/ReadVariableOp_2,sequential/lstm/lstm_cell_8/ReadVariableOp_22\
,sequential/lstm/lstm_cell_8/ReadVariableOp_3,sequential/lstm/lstm_cell_8/ReadVariableOp_32d
0sequential/lstm/lstm_cell_8/split/ReadVariableOp0sequential/lstm/lstm_cell_8/split/ReadVariableOp2h
2sequential/lstm/lstm_cell_8/split_1/ReadVariableOp2sequential/lstm/lstm_cell_8/split_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_74399

inputs$
embedding_74073:???

lstm_74374:
??

lstm_74376:	?

lstm_74378:
??
dense_74393:	?
dense_74395:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_74073*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_74072?
!spatial_dropout1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74080?
lstm/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout1d/PartitionedCall:output:0
lstm_74374
lstm_74376
lstm_74378*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74373?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_74393dense_74395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_74392u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
lstm_while_cond_75220&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_75220___redundant_placeholder0=
9lstm_while_lstm_while_cond_75220___redundant_placeholder1=
9lstm_while_lstm_while_cond_75220___redundant_placeholder2=
9lstm_while_lstm_while_cond_75220___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
ߋ
?	
while_body_76106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_75850

inputs+
embedding_lookup_75844:???
identity??embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding_lookupResourceGatherembedding_lookup_75844Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/75844*-
_output_shapes
:???????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/75844*-
_output_shapes
:????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?<
?
?__inference_lstm_layer_call_and_return_conditional_losses_73713

inputs%
lstm_cell_8_73632:
?? 
lstm_cell_8_73634:	?%
lstm_cell_8_73636:
??
identity??#lstm_cell_8/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_73632lstm_cell_8_73634lstm_cell_8_73636*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73631n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_73632lstm_cell_8_73634lstm_cell_8_73636*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_73645*
condR
while_cond_73644*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????t
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_8_layer_call_fn_77429

inputs
states_0
states_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73631p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_76680
inputs_0=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????^
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????g
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???g
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??Ci
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_4/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_4/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2漂i
$lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_4/GreaterEqualGreaterEqual;lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/CastCast&lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/Mul_1Mullstm_cell_8/dropout_4/Mul:z:0lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_5/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_5/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?޹i
$lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_5/GreaterEqualGreaterEqual;lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/CastCast&lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/Mul_1Mullstm_cell_8/dropout_5/Mul:z:0lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_6/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_6/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_6/GreaterEqualGreaterEqual;lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/CastCast&lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/Mul_1Mullstm_cell_8/dropout_6/Mul:z:0lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_7/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_7/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_7/GreaterEqualGreaterEqual;lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/CastCast&lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/Mul_1Mullstm_cell_8/dropout_7/Mul:z:0lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_4Mulzeros:output:0lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_5Mulzeros:output:0lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_6Mulzeros:output:0lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_7Mulzeros:output:0lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_76462*
condR
while_cond_76461*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_76105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_76105___redundant_placeholder03
/while_while_cond_76105___redundant_placeholder13
/while_while_cond_76105___redundant_placeholder23
/while_while_cond_76105___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?	
while_body_76462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????d
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????s
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???m
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_4/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_4/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??o
*while/lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_4/CastCast,while/lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_4/Mul_1Mul#while/lstm_cell_8/dropout_4/Mul:z:0$while/lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_5/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_5/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??o
*while/lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_5/CastCast,while/lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_5/Mul_1Mul#while/lstm_cell_8/dropout_5/Mul:z:0$while/lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_6/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_6/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_6/CastCast,while/lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_6/Mul_1Mul#while/lstm_cell_8/dropout_6/Mul:z:0$while/lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_7/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_7/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??Ro
*while/lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_7/CastCast,while/lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_7/Mul_1Mul#while/lstm_cell_8/dropout_7/Mul:z:0$while/lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2%while/lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2%while/lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2%while/lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2%while/lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
%__inference_dense_layer_call_fn_77401

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_74392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77716

inputs
states_0
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ɗ?]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??b]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ِ?]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????X
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????^
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????^
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????^
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????^
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?X
Mul_8Muladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????^
Add_1AddV2	Mul_8:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_9Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_9:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????_
mul_10Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????]
mul_11Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????Y
add_5AddV2
mul_10:z:0
mul_11:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
Mul_12Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????_
Add_7AddV2
Mul_12:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????a
mul_13Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????\

Identity_1Identity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_74638
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_74638___redundant_placeholder03
/while_while_cond_74638___redundant_placeholder13
/while_while_cond_74638___redundant_placeholder23
/while_while_cond_74638___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
*__inference_sequential_layer_call_fn_74414
embedding_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_74072

inputs+
embedding_lookup_74066:???
identity??embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding_lookupResourceGatherembedding_lookup_74066Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/74066*-
_output_shapes
:???????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/74066*-
_output_shapes
:????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????y
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*-
_output_shapes
:???????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_77392

inputs=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????^
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????g
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???g
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_4/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_4/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_4/GreaterEqualGreaterEqual;lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/CastCast&lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/Mul_1Mullstm_cell_8/dropout_4/Mul:z:0lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_5/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_5/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2i
$lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_5/GreaterEqualGreaterEqual;lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/CastCast&lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/Mul_1Mullstm_cell_8/dropout_5/Mul:z:0lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_6/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_6/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??Ai
$lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_6/GreaterEqualGreaterEqual;lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/CastCast&lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/Mul_1Mullstm_cell_8/dropout_6/Mul:z:0lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_7/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_7/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?دi
$lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_7/GreaterEqualGreaterEqual;lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/CastCast&lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/Mul_1Mullstm_cell_8/dropout_7/Mul:z:0lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_4Mulzeros:output:0lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_5Mulzeros:output:0lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_6Mulzeros:output:0lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_7Mulzeros:output:0lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_77174*
condR
while_cond_77173*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_74945

inputs$
embedding_74928:???

lstm_74932:
??

lstm_74934:	?

lstm_74936:
??
dense_74939:	?
dense_74941:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?lstm/StatefulPartitionedCall?)spatial_dropout1d/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_74928*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_74072?
)spatial_dropout1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74895?
lstm/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout1d/StatefulPartitionedCall:output:0
lstm_74932
lstm_74934
lstm_74936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74857?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_74939dense_74941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_74392u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*^spatial_dropout1d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2V
)spatial_dropout1d/StatefulPartitionedCall)spatial_dropout1d/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ߋ
?	
while_body_76818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_77173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77173___redundant_placeholder03
/while_while_cond_77173___redundant_placeholder13
/while_while_cond_77173___redundant_placeholder23
/while_while_cond_77173___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?

lstm_while_body_75608&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
6lstm_while_lstm_cell_8_split_readvariableop_resource_0:
??G
8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0:	?D
0lstm_while_lstm_cell_8_readvariableop_resource_0:
??
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
4lstm_while_lstm_cell_8_split_readvariableop_resource:
??E
6lstm_while_lstm_cell_8_split_1_readvariableop_resource:	?B
.lstm_while_lstm_cell_8_readvariableop_resource:
????%lstm/while/lstm_cell_8/ReadVariableOp?'lstm/while/lstm_cell_8/ReadVariableOp_1?'lstm/while/lstm_cell_8/ReadVariableOp_2?'lstm/while/lstm_cell_8/ReadVariableOp_3?+lstm/while/lstm_cell_8/split/ReadVariableOp?-lstm/while/lstm_cell_8/split_1/ReadVariableOp?
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
&lstm/while/lstm_cell_8/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:k
&lstm/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm/while/lstm_cell_8/ones_likeFill/lstm/while/lstm_cell_8/ones_like/Shape:output:0/lstm/while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????i
$lstm/while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm/while/lstm_cell_8/dropout/MulMul)lstm/while/lstm_cell_8/ones_like:output:0-lstm/while/lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????}
$lstm/while/lstm_cell_8/dropout/ShapeShape)lstm/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
;lstm/while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???r
-lstm/while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
+lstm/while/lstm_cell_8/dropout/GreaterEqualGreaterEqualDlstm/while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:06lstm/while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
#lstm/while/lstm_cell_8/dropout/CastCast/lstm/while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
$lstm/while/lstm_cell_8/dropout/Mul_1Mul&lstm/while/lstm_cell_8/dropout/Mul:z:0'lstm/while/lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_1/MulMul)lstm/while/lstm_cell_8/ones_like:output:0/lstm/while/lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????
&lstm/while/lstm_cell_8/dropout_1/ShapeShape)lstm/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???t
/lstm/while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_1/CastCast1lstm/while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_1/Mul_1Mul(lstm/while/lstm_cell_8/dropout_1/Mul:z:0)lstm/while/lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_2/MulMul)lstm/while/lstm_cell_8/ones_like:output:0/lstm/while/lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????
&lstm/while/lstm_cell_8/dropout_2/ShapeShape)lstm/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??0t
/lstm/while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_2/CastCast1lstm/while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_2/Mul_1Mul(lstm/while/lstm_cell_8/dropout_2/Mul:z:0)lstm/while/lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_3/MulMul)lstm/while/lstm_cell_8/ones_like:output:0/lstm/while/lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????
&lstm/while/lstm_cell_8/dropout_3/ShapeShape)lstm/while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???t
/lstm/while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_3/CastCast1lstm/while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_3/Mul_1Mul(lstm/while/lstm_cell_8/dropout_3/Mul:z:0)lstm/while/lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????p
(lstm/while/lstm_cell_8/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:m
(lstm/while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm/while/lstm_cell_8/ones_like_1Fill1lstm/while/lstm_cell_8/ones_like_1/Shape:output:01lstm/while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_4/MulMul+lstm/while/lstm_cell_8/ones_like_1:output:0/lstm/while/lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_4/ShapeShape+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2馼t
/lstm/while/lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_4/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_4/CastCast1lstm/while/lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_4/Mul_1Mul(lstm/while/lstm_cell_8/dropout_4/Mul:z:0)lstm/while/lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_5/MulMul+lstm/while/lstm_cell_8/ones_like_1:output:0/lstm/while/lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_5/ShapeShape+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???t
/lstm/while/lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_5/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_5/CastCast1lstm/while/lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_5/Mul_1Mul(lstm/while/lstm_cell_8/dropout_5/Mul:z:0)lstm/while/lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_6/MulMul+lstm/while/lstm_cell_8/ones_like_1:output:0/lstm/while/lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_6/ShapeShape+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???t
/lstm/while/lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_6/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_6/CastCast1lstm/while/lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_6/Mul_1Mul(lstm/while/lstm_cell_8/dropout_6/Mul:z:0)lstm/while/lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm/while/lstm_cell_8/dropout_7/MulMul+lstm/while/lstm_cell_8/ones_like_1:output:0/lstm/while/lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_7/ShapeShape+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
=lstm/while/lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform/lstm/while/lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???t
/lstm/while/lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm/while/lstm_cell_8/dropout_7/GreaterEqualGreaterEqualFlstm/while/lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:08lstm/while/lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/dropout_7/CastCast1lstm/while/lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
&lstm/while/lstm_cell_8/dropout_7/Mul_1Mul(lstm/while/lstm_cell_8/dropout_7/Mul:z:0)lstm/while/lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0(lstm/while/lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm/while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????h
&lstm/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+lstm/while/lstm_cell_8/split/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
lstm/while/lstm_cell_8/splitSplit/lstm/while/lstm_cell_8/split/split_dim:output:03lstm/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm/while/lstm_cell_8/MatMulMatMullstm/while/lstm_cell_8/mul:z:0%lstm/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_1MatMul lstm/while/lstm_cell_8/mul_1:z:0%lstm/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_2MatMul lstm/while/lstm_cell_8/mul_2:z:0%lstm/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_3MatMul lstm/while/lstm_cell_8/mul_3:z:0%lstm/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????j
(lstm/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-lstm/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
lstm/while/lstm_cell_8/split_1Split1lstm/while/lstm_cell_8/split_1/split_dim:output:05lstm/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm/while/lstm_cell_8/BiasAddBiasAdd'lstm/while/lstm_cell_8/MatMul:product:0'lstm/while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_1BiasAdd)lstm/while/lstm_cell_8/MatMul_1:product:0'lstm/while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_2BiasAdd)lstm/while/lstm_cell_8/MatMul_2:product:0'lstm/while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_3BiasAdd)lstm/while/lstm_cell_8/MatMul_3:product:0'lstm/while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_4Mullstm_while_placeholder_2*lstm/while/lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_5Mullstm_while_placeholder_2*lstm/while/lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_6Mullstm_while_placeholder_2*lstm/while/lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_7Mullstm_while_placeholder_2*lstm/while/lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/ReadVariableOpReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0{
*lstm/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   }
,lstm/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
$lstm/while/lstm_cell_8/strided_sliceStridedSlice-lstm/while/lstm_cell_8/ReadVariableOp:value:03lstm/while/lstm_cell_8/strided_slice/stack:output:05lstm/while/lstm_cell_8/strided_slice/stack_1:output:05lstm/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_4MatMul lstm/while/lstm_cell_8/mul_4:z:0-lstm/while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/addAddV2'lstm/while/lstm_cell_8/BiasAdd:output:0)lstm/while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????a
lstm/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_8Mullstm/while/lstm_cell_8/add:z:0%lstm/while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_1AddV2 lstm/while/lstm_cell_8/Mul_8:z:0'lstm/while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????s
.lstm/while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
,lstm/while/lstm_cell_8/clip_by_value/MinimumMinimum lstm/while/lstm_cell_8/Add_1:z:07lstm/while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
$lstm/while/lstm_cell_8/clip_by_valueMaximum0lstm/while/lstm_cell_8/clip_by_value/Minimum:z:0/lstm/while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   
.lstm/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_1StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_1:value:05lstm/while/lstm_cell_8/strided_slice_1/stack:output:07lstm/while/lstm_cell_8/strided_slice_1/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_5MatMul lstm/while/lstm_cell_8/mul_5:z:0/lstm/while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_2AddV2)lstm/while/lstm_cell_8/BiasAdd_1:output:0)lstm/while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????c
lstm/while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_9Mul lstm/while/lstm_cell_8/add_2:z:0'lstm/while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_3AddV2 lstm/while/lstm_cell_8/Mul_9:z:0'lstm/while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????u
0lstm/while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm/while/lstm_cell_8/clip_by_value_1/MinimumMinimum lstm/while/lstm_cell_8/Add_3:z:09lstm/while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm/while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm/while/lstm_cell_8/clip_by_value_1Maximum2lstm/while/lstm_cell_8/clip_by_value_1/Minimum:z:01lstm/while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_10Mul*lstm/while/lstm_cell_8/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_2StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_2:value:05lstm/while/lstm_cell_8/strided_slice_2/stack:output:07lstm/while/lstm_cell_8/strided_slice_2/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_6MatMul lstm/while/lstm_cell_8/mul_6:z:0/lstm/while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_4AddV2)lstm/while/lstm_cell_8/BiasAdd_2:output:0)lstm/while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????x
lstm/while/lstm_cell_8/TanhTanh lstm/while/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_11Mul(lstm/while/lstm_cell_8/clip_by_value:z:0lstm/while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_5AddV2!lstm/while/lstm_cell_8/mul_10:z:0!lstm/while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_3StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_3:value:05lstm/while/lstm_cell_8/strided_slice_3/stack:output:07lstm/while/lstm_cell_8/strided_slice_3/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_7MatMul lstm/while/lstm_cell_8/mul_7:z:0/lstm/while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_6AddV2)lstm/while/lstm_cell_8/BiasAdd_3:output:0)lstm/while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????c
lstm/while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_12Mul lstm/while/lstm_cell_8/add_6:z:0'lstm/while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_7AddV2!lstm/while/lstm_cell_8/Mul_12:z:0'lstm/while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????u
0lstm/while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm/while/lstm_cell_8/clip_by_value_2/MinimumMinimum lstm/while/lstm_cell_8/Add_7:z:09lstm/while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm/while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm/while/lstm_cell_8/clip_by_value_2Maximum2lstm/while/lstm_cell_8/clip_by_value_2/Minimum:z:01lstm/while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????z
lstm/while/lstm_cell_8/Tanh_1Tanh lstm/while/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_13Mul*lstm/while/lstm_cell_8/clip_by_value_2:z:0!lstm/while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholder!lstm/while/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ?
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: ?
lstm/while/Identity_4Identity!lstm/while/lstm_cell_8/mul_13:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:???????????
lstm/while/Identity_5Identity lstm/while/lstm_cell_8/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:???????????
lstm/while/NoOpNoOp&^lstm/while/lstm_cell_8/ReadVariableOp(^lstm/while/lstm_cell_8/ReadVariableOp_1(^lstm/while/lstm_cell_8/ReadVariableOp_2(^lstm/while/lstm_cell_8/ReadVariableOp_3,^lstm/while/lstm_cell_8/split/ReadVariableOp.^lstm/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"b
.lstm_while_lstm_cell_8_readvariableop_resource0lstm_while_lstm_cell_8_readvariableop_resource_0"r
6lstm_while_lstm_cell_8_split_1_readvariableop_resource8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0"n
4lstm_while_lstm_cell_8_split_readvariableop_resource6lstm_while_lstm_cell_8_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"?
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2N
%lstm/while/lstm_cell_8/ReadVariableOp%lstm/while/lstm_cell_8/ReadVariableOp2R
'lstm/while/lstm_cell_8/ReadVariableOp_1'lstm/while/lstm_cell_8/ReadVariableOp_12R
'lstm/while/lstm_cell_8/ReadVariableOp_2'lstm/while/lstm_cell_8/ReadVariableOp_22R
'lstm/while/lstm_cell_8/ReadVariableOp_3'lstm/while/lstm_cell_8/ReadVariableOp_32Z
+lstm/while/lstm_cell_8/split/ReadVariableOp+lstm/while/lstm_cell_8/split/ReadVariableOp2^
-lstm/while/lstm_cell_8/split_1/ReadVariableOp-lstm/while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
j
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75902

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
k
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74895

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
lstm_while_cond_75607&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_75607___redundant_placeholder0=
9lstm_while_lstm_while_cond_75607___redundant_placeholder1=
9lstm_while_lstm_while_cond_75607___redundant_placeholder2=
9lstm_while_lstm_while_cond_75607___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
*__inference_sequential_layer_call_fn_74977
embedding_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?
?
while_cond_76461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_76461___redundant_placeholder03
/while_while_cond_76461___redundant_placeholder13
/while_while_cond_76461___redundant_placeholder23
/while_while_cond_76461___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
@__inference_dense_layer_call_and_return_conditional_losses_77412

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
Ț
?
?__inference_lstm_layer_call_and_return_conditional_losses_74373

inputs=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_4Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_5Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_6Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_7Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_74219*
condR
while_cond_74218*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_lstm_layer_call_fn_75968

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74857p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_embedding_layer_call_fn_75840

inputs
unknown:???
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_74072u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
j
1__inference_spatial_dropout1d_layer_call_fn_75870

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74895u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_75833

inputs5
 embedding_embedding_lookup_75386:???B
.lstm_lstm_cell_8_split_readvariableop_resource:
???
0lstm_lstm_cell_8_split_1_readvariableop_resource:	?<
(lstm_lstm_cell_8_readvariableop_resource:
??7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookup?lstm/lstm_cell_8/ReadVariableOp?!lstm/lstm_cell_8/ReadVariableOp_1?!lstm/lstm_cell_8/ReadVariableOp_2?!lstm/lstm_cell_8/ReadVariableOp_3?%lstm/lstm_cell_8/split/ReadVariableOp?'lstm/lstm_cell_8/split_1/ReadVariableOp?
lstm/while`
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_75386embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/75386*-
_output_shapes
:???????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/75386*-
_output_shapes
:????????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:???????????u
spatial_dropout1d/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:o
%spatial_dropout1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'spatial_dropout1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'spatial_dropout1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
spatial_dropout1d/strided_sliceStridedSlice spatial_dropout1d/Shape:output:0.spatial_dropout1d/strided_slice/stack:output:00spatial_dropout1d/strided_slice/stack_1:output:00spatial_dropout1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
'spatial_dropout1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)spatial_dropout1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)spatial_dropout1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!spatial_dropout1d/strided_slice_1StridedSlice spatial_dropout1d/Shape:output:00spatial_dropout1d/strided_slice_1/stack:output:02spatial_dropout1d/strided_slice_1/stack_1:output:02spatial_dropout1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
spatial_dropout1d/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
spatial_dropout1d/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0(spatial_dropout1d/dropout/Const:output:0*
T0*-
_output_shapes
:???????????r
0spatial_dropout1d/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.spatial_dropout1d/dropout/random_uniform/shapePack(spatial_dropout1d/strided_slice:output:09spatial_dropout1d/dropout/random_uniform/shape/1:output:0*spatial_dropout1d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
6spatial_dropout1d/dropout/random_uniform/RandomUniformRandomUniform7spatial_dropout1d/dropout/random_uniform/shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0m
(spatial_dropout1d/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&spatial_dropout1d/dropout/GreaterEqualGreaterEqual?spatial_dropout1d/dropout/random_uniform/RandomUniform:output:01spatial_dropout1d/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:???????????
spatial_dropout1d/dropout/CastCast*spatial_dropout1d/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:???????????
spatial_dropout1d/dropout/Mul_1Mul!spatial_dropout1d/dropout/Mul:z:0"spatial_dropout1d/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????]

lstm/ShapeShape#spatial_dropout1d/dropout/Mul_1:z:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: V
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????U
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose	Transpose#spatial_dropout1d/dropout/Mul_1:z:0lstm/transpose/perm:output:0*
T0*-
_output_shapes
:???????????N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskm
 lstm/lstm_cell_8/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:e
 lstm/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/ones_likeFill)lstm/lstm_cell_8/ones_like/Shape:output:0)lstm/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????c
lstm/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout/MulMul#lstm/lstm_cell_8/ones_like:output:0'lstm/lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????q
lstm/lstm_cell_8/dropout/ShapeShape#lstm/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
5lstm/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ڀl
'lstm/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
%lstm/lstm_cell_8/dropout/GreaterEqualGreaterEqual>lstm/lstm_cell_8/dropout/random_uniform/RandomUniform:output:00lstm/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout/CastCast)lstm/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout/Mul_1Mul lstm/lstm_cell_8/dropout/Mul:z:0!lstm/lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_1/MulMul#lstm/lstm_cell_8/ones_like:output:0)lstm/lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????s
 lstm/lstm_cell_8/dropout_1/ShapeShape#lstm/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_1/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_1/CastCast+lstm/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_1/Mul_1Mul"lstm/lstm_cell_8/dropout_1/Mul:z:0#lstm/lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_2/MulMul#lstm/lstm_cell_8/ones_like:output:0)lstm/lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????s
 lstm/lstm_cell_8/dropout_2/ShapeShape#lstm/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_2/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_2/CastCast+lstm/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_2/Mul_1Mul"lstm/lstm_cell_8/dropout_2/Mul:z:0#lstm/lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_3/MulMul#lstm/lstm_cell_8/ones_like:output:0)lstm/lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????s
 lstm/lstm_cell_8/dropout_3/ShapeShape#lstm/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_3/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_3/CastCast+lstm/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_3/Mul_1Mul"lstm/lstm_cell_8/dropout_3/Mul:z:0#lstm/lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????e
"lstm/lstm_cell_8/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:g
"lstm/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/ones_like_1Fill+lstm/lstm_cell_8/ones_like_1/Shape:output:0+lstm/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_4/MulMul%lstm/lstm_cell_8/ones_like_1:output:0)lstm/lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????u
 lstm/lstm_cell_8/dropout_4/ShapeShape%lstm/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_4/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_4/CastCast+lstm/lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_4/Mul_1Mul"lstm/lstm_cell_8/dropout_4/Mul:z:0#lstm/lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_5/MulMul%lstm/lstm_cell_8/ones_like_1:output:0)lstm/lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????u
 lstm/lstm_cell_8/dropout_5/ShapeShape%lstm/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_5/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_5/CastCast+lstm/lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_5/Mul_1Mul"lstm/lstm_cell_8/dropout_5/Mul:z:0#lstm/lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_6/MulMul%lstm/lstm_cell_8/ones_like_1:output:0)lstm/lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????u
 lstm/lstm_cell_8/dropout_6/ShapeShape%lstm/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2鉔n
)lstm/lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_6/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_6/CastCast+lstm/lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_6/Mul_1Mul"lstm/lstm_cell_8/dropout_6/Mul:z:0#lstm/lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/dropout_7/MulMul%lstm/lstm_cell_8/ones_like_1:output:0)lstm/lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????u
 lstm/lstm_cell_8/dropout_7/ShapeShape%lstm/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
7lstm/lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform)lstm/lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???n
)lstm/lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm/lstm_cell_8/dropout_7/GreaterEqualGreaterEqual@lstm/lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:02lstm/lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/dropout_7/CastCast+lstm/lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
 lstm/lstm_cell_8/dropout_7/Mul_1Mul"lstm/lstm_cell_8/dropout_7/Mul:z:0#lstm/lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mulMullstm/strided_slice_2:output:0"lstm/lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_1Mullstm/strided_slice_2:output:0$lstm/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_2Mullstm/strided_slice_2:output:0$lstm/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_3Mullstm/strided_slice_2:output:0$lstm/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????b
 lstm/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%lstm/lstm_cell_8/split/ReadVariableOpReadVariableOp.lstm_lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm/lstm_cell_8/splitSplit)lstm/lstm_cell_8/split/split_dim:output:0-lstm/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm/lstm_cell_8/MatMulMatMullstm/lstm_cell_8/mul:z:0lstm/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_1MatMullstm/lstm_cell_8/mul_1:z:0lstm/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_2MatMullstm/lstm_cell_8/mul_2:z:0lstm/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_3MatMullstm/lstm_cell_8/mul_3:z:0lstm/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????d
"lstm/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'lstm/lstm_cell_8/split_1/ReadVariableOpReadVariableOp0lstm_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm/lstm_cell_8/split_1Split+lstm/lstm_cell_8/split_1/split_dim:output:0/lstm/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm/lstm_cell_8/BiasAddBiasAdd!lstm/lstm_cell_8/MatMul:product:0!lstm/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_1BiasAdd#lstm/lstm_cell_8/MatMul_1:product:0!lstm/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_2BiasAdd#lstm/lstm_cell_8/MatMul_2:product:0!lstm/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_3BiasAdd#lstm/lstm_cell_8/MatMul_3:product:0!lstm/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_4Mullstm/zeros:output:0$lstm/lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_5Mullstm/zeros:output:0$lstm/lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_6Mullstm/zeros:output:0$lstm/lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_7Mullstm/zeros:output:0$lstm/lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/ReadVariableOpReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0u
$lstm/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   w
&lstm/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm/lstm_cell_8/strided_sliceStridedSlice'lstm/lstm_cell_8/ReadVariableOp:value:0-lstm/lstm_cell_8/strided_slice/stack:output:0/lstm/lstm_cell_8/strided_slice/stack_1:output:0/lstm/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_4MatMullstm/lstm_cell_8/mul_4:z:0'lstm/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/addAddV2!lstm/lstm_cell_8/BiasAdd:output:0#lstm/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????[
lstm/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_8Mullstm/lstm_cell_8/add:z:0lstm/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_1AddV2lstm/lstm_cell_8/Mul_8:z:0!lstm/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????m
(lstm/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm/lstm_cell_8/clip_by_value/MinimumMinimumlstm/lstm_cell_8/Add_1:z:01lstm/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/lstm_cell_8/clip_by_valueMaximum*lstm/lstm_cell_8/clip_by_value/Minimum:z:0)lstm/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_1ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   y
(lstm/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_1StridedSlice)lstm/lstm_cell_8/ReadVariableOp_1:value:0/lstm/lstm_cell_8/strided_slice_1/stack:output:01lstm/lstm_cell_8/strided_slice_1/stack_1:output:01lstm/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_5MatMullstm/lstm_cell_8/mul_5:z:0)lstm/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_2AddV2#lstm/lstm_cell_8/BiasAdd_1:output:0#lstm/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????]
lstm/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_9Mullstm/lstm_cell_8/add_2:z:0!lstm/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_3AddV2lstm/lstm_cell_8/Mul_9:z:0!lstm/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????o
*lstm/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm/lstm_cell_8/clip_by_value_1/MinimumMinimumlstm/lstm_cell_8/Add_3:z:03lstm/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm/lstm_cell_8/clip_by_value_1Maximum,lstm/lstm_cell_8/clip_by_value_1/Minimum:z:0+lstm/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_10Mul$lstm/lstm_cell_8/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_2ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_2StridedSlice)lstm/lstm_cell_8/ReadVariableOp_2:value:0/lstm/lstm_cell_8/strided_slice_2/stack:output:01lstm/lstm_cell_8/strided_slice_2/stack_1:output:01lstm/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_6MatMullstm/lstm_cell_8/mul_6:z:0)lstm/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_4AddV2#lstm/lstm_cell_8/BiasAdd_2:output:0#lstm/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????l
lstm/lstm_cell_8/TanhTanhlstm/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_11Mul"lstm/lstm_cell_8/clip_by_value:z:0lstm/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_5AddV2lstm/lstm_cell_8/mul_10:z:0lstm/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_3ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(lstm/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_3StridedSlice)lstm/lstm_cell_8/ReadVariableOp_3:value:0/lstm/lstm_cell_8/strided_slice_3/stack:output:01lstm/lstm_cell_8/strided_slice_3/stack_1:output:01lstm/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_7MatMullstm/lstm_cell_8/mul_7:z:0)lstm/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_6AddV2#lstm/lstm_cell_8/BiasAdd_3:output:0#lstm/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????]
lstm/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_12Mullstm/lstm_cell_8/add_6:z:0!lstm/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_7AddV2lstm/lstm_cell_8/Mul_12:z:0!lstm/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????o
*lstm/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm/lstm_cell_8/clip_by_value_2/MinimumMinimumlstm/lstm_cell_8/Add_7:z:03lstm/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm/lstm_cell_8/clip_by_value_2Maximum,lstm/lstm_cell_8/clip_by_value_2/Minimum:z:0+lstm/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????n
lstm/lstm_cell_8/Tanh_1Tanhlstm/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_13Mul$lstm/lstm_cell_8/clip_by_value_2:z:0lstm/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_lstm_cell_8_split_readvariableop_resource0lstm_lstm_cell_8_split_1_readvariableop_resource(lstm_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_75608*!
condR
lstm_while_cond_75607*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup ^lstm/lstm_cell_8/ReadVariableOp"^lstm/lstm_cell_8/ReadVariableOp_1"^lstm/lstm_cell_8/ReadVariableOp_2"^lstm/lstm_cell_8/ReadVariableOp_3&^lstm/lstm_cell_8/split/ReadVariableOp(^lstm/lstm_cell_8/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2B
lstm/lstm_cell_8/ReadVariableOplstm/lstm_cell_8/ReadVariableOp2F
!lstm/lstm_cell_8/ReadVariableOp_1!lstm/lstm_cell_8/ReadVariableOp_12F
!lstm/lstm_cell_8/ReadVariableOp_2!lstm/lstm_cell_8/ReadVariableOp_22F
!lstm/lstm_cell_8/ReadVariableOp_3!lstm/lstm_cell_8/ReadVariableOp_32N
%lstm/lstm_cell_8/split/ReadVariableOp%lstm/lstm_cell_8/split/ReadVariableOp2R
'lstm/lstm_cell_8/split_1/ReadVariableOp'lstm/lstm_cell_8/split_1/ReadVariableOp2

lstm/while
lstm/while:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77549

inputs
states_0
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????Y
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????_
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????_
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????_
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????_
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?X
Mul_8Muladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????^
Add_1AddV2	Mul_8:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_9Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_9:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????_
mul_10Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????]
mul_11Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????Y
add_5AddV2
mul_10:z:0
mul_11:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
Mul_12Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????_
Add_7AddV2
Mul_12:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????a
mul_13Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????\

Identity_1Identity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
j
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75875

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_76260
inputs_0=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_4Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_5Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_6Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_7Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_76106*
condR
while_cond_76105*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
 sequential_lstm_while_cond_73284<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1S
Osequential_lstm_while_sequential_lstm_while_cond_73284___redundant_placeholder0S
Osequential_lstm_while_sequential_lstm_while_cond_73284___redundant_placeholder1S
Osequential_lstm_while_sequential_lstm_while_cond_73284___redundant_placeholder2S
Osequential_lstm_while_sequential_lstm_while_cond_73284___redundant_placeholder3"
sequential_lstm_while_identity
?
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: k
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?"
?
while_body_73645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_73669_0:
??(
while_lstm_cell_8_73671_0:	?-
while_lstm_cell_8_73673_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_73669:
??&
while_lstm_cell_8_73671:	?+
while_lstm_cell_8_73673:
????)while/lstm_cell_8/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_73669_0while_lstm_cell_8_73671_0while_lstm_cell_8_73673_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73631?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_8_73669while_lstm_cell_8_73669_0"4
while_lstm_cell_8_73671while_lstm_cell_8_73671_0"4
while_lstm_cell_8_73673while_lstm_cell_8_73673_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_75382

inputs5
 embedding_embedding_lookup_75080:???B
.lstm_lstm_cell_8_split_readvariableop_resource:
???
0lstm_lstm_cell_8_split_1_readvariableop_resource:	?<
(lstm_lstm_cell_8_readvariableop_resource:
??7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookup?lstm/lstm_cell_8/ReadVariableOp?!lstm/lstm_cell_8/ReadVariableOp_1?!lstm/lstm_cell_8/ReadVariableOp_2?!lstm/lstm_cell_8/ReadVariableOp_3?%lstm/lstm_cell_8/split/ReadVariableOp?'lstm/lstm_cell_8/split_1/ReadVariableOp?
lstm/while`
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_75080embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/75080*-
_output_shapes
:???????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/75080*-
_output_shapes
:????????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
spatial_dropout1d/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:???????????]

lstm/ShapeShape#spatial_dropout1d/Identity:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: V
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:??????????U
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: X
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose	Transpose#spatial_dropout1d/Identity:output:0lstm/transpose/perm:output:0*
T0*-
_output_shapes
:???????????N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskm
 lstm/lstm_cell_8/ones_like/ShapeShapelstm/strided_slice_2:output:0*
T0*
_output_shapes
:e
 lstm/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/ones_likeFill)lstm/lstm_cell_8/ones_like/Shape:output:0)lstm/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????e
"lstm/lstm_cell_8/ones_like_1/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:g
"lstm/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm/lstm_cell_8/ones_like_1Fill+lstm/lstm_cell_8/ones_like_1/Shape:output:0+lstm/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mulMullstm/strided_slice_2:output:0#lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_1Mullstm/strided_slice_2:output:0#lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_2Mullstm/strided_slice_2:output:0#lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_3Mullstm/strided_slice_2:output:0#lstm/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????b
 lstm/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
%lstm/lstm_cell_8/split/ReadVariableOpReadVariableOp.lstm_lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm/lstm_cell_8/splitSplit)lstm/lstm_cell_8/split/split_dim:output:0-lstm/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm/lstm_cell_8/MatMulMatMullstm/lstm_cell_8/mul:z:0lstm/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_1MatMullstm/lstm_cell_8/mul_1:z:0lstm/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_2MatMullstm/lstm_cell_8/mul_2:z:0lstm/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/MatMul_3MatMullstm/lstm_cell_8/mul_3:z:0lstm/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????d
"lstm/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'lstm/lstm_cell_8/split_1/ReadVariableOpReadVariableOp0lstm_lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm/lstm_cell_8/split_1Split+lstm/lstm_cell_8/split_1/split_dim:output:0/lstm/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm/lstm_cell_8/BiasAddBiasAdd!lstm/lstm_cell_8/MatMul:product:0!lstm/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_1BiasAdd#lstm/lstm_cell_8/MatMul_1:product:0!lstm/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_2BiasAdd#lstm/lstm_cell_8/MatMul_2:product:0!lstm/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/BiasAdd_3BiasAdd#lstm/lstm_cell_8/MatMul_3:product:0!lstm/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_4Mullstm/zeros:output:0%lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_5Mullstm/zeros:output:0%lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_6Mullstm/zeros:output:0%lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_7Mullstm/zeros:output:0%lstm/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/ReadVariableOpReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0u
$lstm/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   w
&lstm/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm/lstm_cell_8/strided_sliceStridedSlice'lstm/lstm_cell_8/ReadVariableOp:value:0-lstm/lstm_cell_8/strided_slice/stack:output:0/lstm/lstm_cell_8/strided_slice/stack_1:output:0/lstm/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_4MatMullstm/lstm_cell_8/mul_4:z:0'lstm/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/addAddV2!lstm/lstm_cell_8/BiasAdd:output:0#lstm/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????[
lstm/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_8Mullstm/lstm_cell_8/add:z:0lstm/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_1AddV2lstm/lstm_cell_8/Mul_8:z:0!lstm/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????m
(lstm/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm/lstm_cell_8/clip_by_value/MinimumMinimumlstm/lstm_cell_8/Add_1:z:01lstm/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????e
 lstm/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm/lstm_cell_8/clip_by_valueMaximum*lstm/lstm_cell_8/clip_by_value/Minimum:z:0)lstm/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_1ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   y
(lstm/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_1StridedSlice)lstm/lstm_cell_8/ReadVariableOp_1:value:0/lstm/lstm_cell_8/strided_slice_1/stack:output:01lstm/lstm_cell_8/strided_slice_1/stack_1:output:01lstm/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_5MatMullstm/lstm_cell_8/mul_5:z:0)lstm/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_2AddV2#lstm/lstm_cell_8/BiasAdd_1:output:0#lstm/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????]
lstm/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_9Mullstm/lstm_cell_8/add_2:z:0!lstm/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_3AddV2lstm/lstm_cell_8/Mul_9:z:0!lstm/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????o
*lstm/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm/lstm_cell_8/clip_by_value_1/MinimumMinimumlstm/lstm_cell_8/Add_3:z:03lstm/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm/lstm_cell_8/clip_by_value_1Maximum,lstm/lstm_cell_8/clip_by_value_1/Minimum:z:0+lstm/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_10Mul$lstm/lstm_cell_8/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_2ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_2StridedSlice)lstm/lstm_cell_8/ReadVariableOp_2:value:0/lstm/lstm_cell_8/strided_slice_2/stack:output:01lstm/lstm_cell_8/strided_slice_2/stack_1:output:01lstm/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_6MatMullstm/lstm_cell_8/mul_6:z:0)lstm/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_4AddV2#lstm/lstm_cell_8/BiasAdd_2:output:0#lstm/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????l
lstm/lstm_cell_8/TanhTanhlstm/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_11Mul"lstm/lstm_cell_8/clip_by_value:z:0lstm/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_5AddV2lstm/lstm_cell_8/mul_10:z:0lstm/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
!lstm/lstm_cell_8/ReadVariableOp_3ReadVariableOp(lstm_lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  y
(lstm/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(lstm/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm/lstm_cell_8/strided_slice_3StridedSlice)lstm/lstm_cell_8/ReadVariableOp_3:value:0/lstm/lstm_cell_8/strided_slice_3/stack:output:01lstm/lstm_cell_8/strided_slice_3/stack_1:output:01lstm/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/lstm_cell_8/MatMul_7MatMullstm/lstm_cell_8/mul_7:z:0)lstm/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/add_6AddV2#lstm/lstm_cell_8/BiasAdd_3:output:0#lstm/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????]
lstm/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>]
lstm/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/lstm_cell_8/Mul_12Mullstm/lstm_cell_8/add_6:z:0!lstm/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/Add_7AddV2lstm/lstm_cell_8/Mul_12:z:0!lstm/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????o
*lstm/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm/lstm_cell_8/clip_by_value_2/MinimumMinimumlstm/lstm_cell_8/Add_7:z:03lstm/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm/lstm_cell_8/clip_by_value_2Maximum,lstm/lstm_cell_8/clip_by_value_2/Minimum:z:0+lstm/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????n
lstm/lstm_cell_8/Tanh_1Tanhlstm/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm/lstm_cell_8/mul_13Mul$lstm/lstm_cell_8/clip_by_value_2:z:0lstm/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_lstm_cell_8_split_readvariableop_resource0lstm_lstm_cell_8_split_1_readvariableop_resource(lstm_lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *!
bodyR
lstm_while_body_75221*!
condR
lstm_while_cond_75220*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*-
_output_shapes
:????????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup ^lstm/lstm_cell_8/ReadVariableOp"^lstm/lstm_cell_8/ReadVariableOp_1"^lstm/lstm_cell_8/ReadVariableOp_2"^lstm/lstm_cell_8/ReadVariableOp_3&^lstm/lstm_cell_8/split/ReadVariableOp(^lstm/lstm_cell_8/split_1/ReadVariableOp^lstm/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2B
lstm/lstm_cell_8/ReadVariableOplstm/lstm_cell_8/ReadVariableOp2F
!lstm/lstm_cell_8/ReadVariableOp_1!lstm/lstm_cell_8/ReadVariableOp_12F
!lstm/lstm_cell_8/ReadVariableOp_2!lstm/lstm_cell_8/ReadVariableOp_22F
!lstm/lstm_cell_8/ReadVariableOp_3!lstm/lstm_cell_8/ReadVariableOp_32N
%lstm/lstm_cell_8/split/ReadVariableOp%lstm/lstm_cell_8/split/ReadVariableOp2R
'lstm/lstm_cell_8/split_1/ReadVariableOp'lstm/lstm_cell_8/split_1/ReadVariableOp2

lstm/while
lstm/while:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_spatial_dropout1d_layer_call_fn_75855

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_73455v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_8_layer_call_fn_77446

inputs
states_0
states_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73911p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?

lstm_while_body_75221&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0J
6lstm_while_lstm_cell_8_split_readvariableop_resource_0:
??G
8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0:	?D
0lstm_while_lstm_cell_8_readvariableop_resource_0:
??
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorH
4lstm_while_lstm_cell_8_split_readvariableop_resource:
??E
6lstm_while_lstm_cell_8_split_1_readvariableop_resource:	?B
.lstm_while_lstm_cell_8_readvariableop_resource:
????%lstm/while/lstm_cell_8/ReadVariableOp?'lstm/while/lstm_cell_8/ReadVariableOp_1?'lstm/while/lstm_cell_8/ReadVariableOp_2?'lstm/while/lstm_cell_8/ReadVariableOp_3?+lstm/while/lstm_cell_8/split/ReadVariableOp?-lstm/while/lstm_cell_8/split_1/ReadVariableOp?
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
&lstm/while/lstm_cell_8/ones_like/ShapeShape5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:k
&lstm/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm/while/lstm_cell_8/ones_likeFill/lstm/while/lstm_cell_8/ones_like/Shape:output:0/lstm/while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????p
(lstm/while/lstm_cell_8/ones_like_1/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:m
(lstm/while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm/while/lstm_cell_8/ones_like_1Fill1lstm/while/lstm_cell_8/ones_like_1/Shape:output:01lstm/while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mulMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_1Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_2Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_3Mul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0)lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????h
&lstm/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
+lstm/while/lstm_cell_8/split/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
lstm/while/lstm_cell_8/splitSplit/lstm/while/lstm_cell_8/split/split_dim:output:03lstm/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm/while/lstm_cell_8/MatMulMatMullstm/while/lstm_cell_8/mul:z:0%lstm/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_1MatMul lstm/while/lstm_cell_8/mul_1:z:0%lstm/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_2MatMul lstm/while/lstm_cell_8/mul_2:z:0%lstm/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/MatMul_3MatMul lstm/while/lstm_cell_8/mul_3:z:0%lstm/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????j
(lstm/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
-lstm/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
lstm/while/lstm_cell_8/split_1Split1lstm/while/lstm_cell_8/split_1/split_dim:output:05lstm/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm/while/lstm_cell_8/BiasAddBiasAdd'lstm/while/lstm_cell_8/MatMul:product:0'lstm/while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_1BiasAdd)lstm/while/lstm_cell_8/MatMul_1:product:0'lstm/while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_2BiasAdd)lstm/while/lstm_cell_8/MatMul_2:product:0'lstm/while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
 lstm/while/lstm_cell_8/BiasAdd_3BiasAdd)lstm/while/lstm_cell_8/MatMul_3:product:0'lstm/while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_4Mullstm_while_placeholder_2+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_5Mullstm_while_placeholder_2+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_6Mullstm_while_placeholder_2+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_7Mullstm_while_placeholder_2+lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
%lstm/while/lstm_cell_8/ReadVariableOpReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0{
*lstm/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   }
,lstm/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
$lstm/while/lstm_cell_8/strided_sliceStridedSlice-lstm/while/lstm_cell_8/ReadVariableOp:value:03lstm/while/lstm_cell_8/strided_slice/stack:output:05lstm/while/lstm_cell_8/strided_slice/stack_1:output:05lstm/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_4MatMul lstm/while/lstm_cell_8/mul_4:z:0-lstm/while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/addAddV2'lstm/while/lstm_cell_8/BiasAdd:output:0)lstm/while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????a
lstm/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_8Mullstm/while/lstm_cell_8/add:z:0%lstm/while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_1AddV2 lstm/while/lstm_cell_8/Mul_8:z:0'lstm/while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????s
.lstm/while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
,lstm/while/lstm_cell_8/clip_by_value/MinimumMinimum lstm/while/lstm_cell_8/Add_1:z:07lstm/while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????k
&lstm/while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
$lstm/while/lstm_cell_8/clip_by_valueMaximum0lstm/while/lstm_cell_8/clip_by_value/Minimum:z:0/lstm/while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   
.lstm/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_1StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_1:value:05lstm/while/lstm_cell_8/strided_slice_1/stack:output:07lstm/while/lstm_cell_8/strided_slice_1/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_5MatMul lstm/while/lstm_cell_8/mul_5:z:0/lstm/while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_2AddV2)lstm/while/lstm_cell_8/BiasAdd_1:output:0)lstm/while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????c
lstm/while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_9Mul lstm/while/lstm_cell_8/add_2:z:0'lstm/while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_3AddV2 lstm/while/lstm_cell_8/Mul_9:z:0'lstm/while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????u
0lstm/while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm/while/lstm_cell_8/clip_by_value_1/MinimumMinimum lstm/while/lstm_cell_8/Add_3:z:09lstm/while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm/while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm/while/lstm_cell_8/clip_by_value_1Maximum2lstm/while/lstm_cell_8/clip_by_value_1/Minimum:z:01lstm/while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_10Mul*lstm/while/lstm_cell_8/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_2StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_2:value:05lstm/while/lstm_cell_8/strided_slice_2/stack:output:07lstm/while/lstm_cell_8/strided_slice_2/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_6MatMul lstm/while/lstm_cell_8/mul_6:z:0/lstm/while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_4AddV2)lstm/while/lstm_cell_8/BiasAdd_2:output:0)lstm/while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????x
lstm/while/lstm_cell_8/TanhTanh lstm/while/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_11Mul(lstm/while/lstm_cell_8/clip_by_value:z:0lstm/while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_5AddV2!lstm/while/lstm_cell_8/mul_10:z:0!lstm/while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
'lstm/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp0lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  
.lstm/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm/while/lstm_cell_8/strided_slice_3StridedSlice/lstm/while/lstm_cell_8/ReadVariableOp_3:value:05lstm/while/lstm_cell_8/strided_slice_3/stack:output:07lstm/while/lstm_cell_8/strided_slice_3/stack_1:output:07lstm/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm/while/lstm_cell_8/MatMul_7MatMul lstm/while/lstm_cell_8/mul_7:z:0/lstm/while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/add_6AddV2)lstm/while/lstm_cell_8/BiasAdd_3:output:0)lstm/while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????c
lstm/while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>c
lstm/while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm/while/lstm_cell_8/Mul_12Mul lstm/while/lstm_cell_8/add_6:z:0'lstm/while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/Add_7AddV2!lstm/while/lstm_cell_8/Mul_12:z:0'lstm/while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????u
0lstm/while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm/while/lstm_cell_8/clip_by_value_2/MinimumMinimum lstm/while/lstm_cell_8/Add_7:z:09lstm/while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm/while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm/while/lstm_cell_8/clip_by_value_2Maximum2lstm/while/lstm_cell_8/clip_by_value_2/Minimum:z:01lstm/while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????z
lstm/while/lstm_cell_8/Tanh_1Tanh lstm/while/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm/while/lstm_cell_8/mul_13Mul*lstm/while/lstm_cell_8/clip_by_value_2:z:0!lstm/while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholder!lstm/while/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ?
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: ?
lstm/while/Identity_4Identity!lstm/while/lstm_cell_8/mul_13:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:???????????
lstm/while/Identity_5Identity lstm/while/lstm_cell_8/add_5:z:0^lstm/while/NoOp*
T0*(
_output_shapes
:???????????
lstm/while/NoOpNoOp&^lstm/while/lstm_cell_8/ReadVariableOp(^lstm/while/lstm_cell_8/ReadVariableOp_1(^lstm/while/lstm_cell_8/ReadVariableOp_2(^lstm/while/lstm_cell_8/ReadVariableOp_3,^lstm/while/lstm_cell_8/split/ReadVariableOp.^lstm/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"b
.lstm_while_lstm_cell_8_readvariableop_resource0lstm_while_lstm_cell_8_readvariableop_resource_0"r
6lstm_while_lstm_cell_8_split_1_readvariableop_resource8lstm_while_lstm_cell_8_split_1_readvariableop_resource_0"n
4lstm_while_lstm_cell_8_split_readvariableop_resource6lstm_while_lstm_cell_8_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"?
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2N
%lstm/while/lstm_cell_8/ReadVariableOp%lstm/while/lstm_cell_8/ReadVariableOp2R
'lstm/while/lstm_cell_8/ReadVariableOp_1'lstm/while/lstm_cell_8/ReadVariableOp_12R
'lstm/while/lstm_cell_8/ReadVariableOp_2'lstm/while/lstm_cell_8/ReadVariableOp_22R
'lstm/while/lstm_cell_8/ReadVariableOp_3'lstm/while/lstm_cell_8/ReadVariableOp_32Z
+lstm/while/lstm_cell_8/split/ReadVariableOp+lstm/while/lstm_cell_8/split/ReadVariableOp2^
-lstm/while/lstm_cell_8/split_1/ReadVariableOp-lstm/while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_73644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_73644___redundant_placeholder03
/while_while_cond_73644___redundant_placeholder13
/while_while_cond_73644___redundant_placeholder23
/while_while_cond_73644___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
k
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_73482

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?	
while_body_77174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????d
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????s
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???m
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ԙo
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_4/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_4/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_4/CastCast,while/lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_4/Mul_1Mul#while/lstm_cell_8/dropout_4/Mul:z:0$while/lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_5/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_5/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2䃊o
*while/lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_5/CastCast,while/lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_5/Mul_1Mul#while/lstm_cell_8/dropout_5/Mul:z:0$while/lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_6/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_6/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_6/CastCast,while/lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_6/Mul_1Mul#while/lstm_cell_8/dropout_6/Mul:z:0$while/lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_7/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_7/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_7/CastCast,while/lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_7/Mul_1Mul#while/lstm_cell_8/dropout_7/Mul:z:0$while/lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2%while/lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2%while/lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2%while/lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2%while/lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_lstm_layer_call_fn_75935
inputs_0
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_73713p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
ߋ
?	
while_body_74219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2&while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_74857

inputs=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????^
lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout/MulMullstm_cell_8/ones_like:output:0"lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????g
lstm_cell_8/dropout/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??g
"lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_8/dropout/GreaterEqualGreaterEqual9lstm_cell_8/dropout/random_uniform/RandomUniform:output:0+lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout/CastCast$lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout/Mul_1Mullstm_cell_8/dropout/Mul:z:0lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_1/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_1/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_1/GreaterEqualGreaterEqual;lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/CastCast&lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_1/Mul_1Mullstm_cell_8/dropout_1/Mul:z:0lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_2/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_2/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_2/GreaterEqualGreaterEqual;lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/CastCast&lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_2/Mul_1Mullstm_cell_8/dropout_2/Mul:z:0lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_3/MulMullstm_cell_8/ones_like:output:0$lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????i
lstm_cell_8/dropout_3/ShapeShapelstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_3/GreaterEqualGreaterEqual;lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/CastCast&lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_3/Mul_1Mullstm_cell_8/dropout_3/Mul:z:0lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_4/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_4/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?֑i
$lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_4/GreaterEqualGreaterEqual;lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/CastCast&lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_4/Mul_1Mullstm_cell_8/dropout_4/Mul:z:0lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_5/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_5/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ƈi
$lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_5/GreaterEqualGreaterEqual;lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/CastCast&lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_5/Mul_1Mullstm_cell_8/dropout_5/Mul:z:0lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_6/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_6/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_6/GreaterEqualGreaterEqual;lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/CastCast&lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_6/Mul_1Mullstm_cell_8/dropout_6/Mul:z:0lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/dropout_7/MulMul lstm_cell_8/ones_like_1:output:0$lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????k
lstm_cell_8/dropout_7/ShapeShape lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_8/dropout_7/GreaterEqualGreaterEqual;lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/CastCast&lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
lstm_cell_8/dropout_7/Mul_1Mullstm_cell_8/dropout_7/Mul:z:0lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_4Mulzeros:output:0lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_5Mulzeros:output:0lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_6Mulzeros:output:0lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????|
lstm_cell_8/mul_7Mulzeros:output:0lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_74639*
condR
while_cond_74638*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
while_cond_74218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_74218___redundant_placeholder03
/while_while_cond_74218___redundant_placeholder13
/while_while_cond_74218___redundant_placeholder23
/while_while_cond_74218___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_74997
embedding_input$
embedding_74980:???

lstm_74984:
??

lstm_74986:	?

lstm_74988:
??
dense_74991:	?
dense_74993:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_74980*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_74072?
!spatial_dropout1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74080?
lstm/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout1d/PartitionedCall:output:0
lstm_74984
lstm_74986
lstm_74988*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74373?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_74991dense_74993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_74392u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_75017
embedding_input$
embedding_75000:???

lstm_75004:
??

lstm_75006:	?

lstm_75008:
??
dense_75011:	?
dense_75013:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?lstm/StatefulPartitionedCall?)spatial_dropout1d/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_75000*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_74072?
)spatial_dropout1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74895?
lstm/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout1d/StatefulPartitionedCall:output:0
lstm_75004
lstm_75006
lstm_75008*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74857?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_75011dense_75013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_74392u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^lstm/StatefulPartitionedCall*^spatial_dropout1d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2V
)spatial_dropout1d/StatefulPartitionedCall)spatial_dropout1d/StatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?<
?
?__inference_lstm_layer_call_and_return_conditional_losses_74046

inputs%
lstm_cell_8_73965:
?? 
lstm_cell_8_73967:	?%
lstm_cell_8_73969:
??
identity??#lstm_cell_8/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_73965lstm_cell_8_73967lstm_cell_8_73969*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73911n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_73965lstm_cell_8_73967lstm_cell_8_73969*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_73978*
condR
while_cond_73977*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????t
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
j
1__inference_spatial_dropout1d_layer_call_fn_75860

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_73482?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
k
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75924

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
Ț
?
?__inference_lstm_layer_call_and_return_conditional_losses_76972

inputs=
)lstm_cell_8_split_readvariableop_resource:
??:
+lstm_cell_8_split_1_readvariableop_resource:	?7
#lstm_cell_8_readvariableop_resource:
??
identity??lstm_cell_8/ReadVariableOp?lstm_cell_8/ReadVariableOp_1?lstm_cell_8/ReadVariableOp_2?lstm_cell_8/ReadVariableOp_3? lstm_cell_8/split/ReadVariableOp?"lstm_cell_8/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: Q
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????P
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          o
	transpose	Transposeinputstranspose/perm:output:0*
T0*-
_output_shapes
:???????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskc
lstm_cell_8/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_likeFill$lstm_cell_8/ones_like/Shape:output:0$lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????[
lstm_cell_8/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_8/ones_like_1Fill&lstm_cell_8/ones_like_1/Shape:output:0&lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mulMulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_1Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_2Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_3Mulstrided_slice_2:output:0lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_8/split/ReadVariableOpReadVariableOp)lstm_cell_8_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0(lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_8/MatMulMatMullstm_cell_8/mul:z:0lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_1MatMullstm_cell_8/mul_1:z:0lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_2MatMullstm_cell_8/mul_2:z:0lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/MatMul_3MatMullstm_cell_8/mul_3:z:0lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_8/split_1/ReadVariableOpReadVariableOp+lstm_cell_8_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_8/split_1Split&lstm_cell_8/split_1/split_dim:output:0*lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_8/BiasAddBiasAddlstm_cell_8/MatMul:product:0lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_1BiasAddlstm_cell_8/MatMul_1:product:0lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_2BiasAddlstm_cell_8/MatMul_2:product:0lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_8/BiasAdd_3BiasAddlstm_cell_8/MatMul_3:product:0lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_4Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_5Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_6Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/mul_7Mulzeros:output:0 lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOpReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_sliceStridedSlice"lstm_cell_8/ReadVariableOp:value:0(lstm_cell_8/strided_slice/stack:output:0*lstm_cell_8/strided_slice/stack_1:output:0*lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_4MatMullstm_cell_8/mul_4:z:0"lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/addAddV2lstm_cell_8/BiasAdd:output:0lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?|
lstm_cell_8/Mul_8Mullstm_cell_8/add:z:0lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_1AddV2lstm_cell_8/Mul_8:z:0lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_8/clip_by_value/MinimumMinimumlstm_cell_8/Add_1:z:0,lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_valueMaximum%lstm_cell_8/clip_by_value/Minimum:z:0$lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_1ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_1StridedSlice$lstm_cell_8/ReadVariableOp_1:value:0*lstm_cell_8/strided_slice_1/stack:output:0,lstm_cell_8/strided_slice_1/stack_1:output:0,lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_5MatMullstm_cell_8/mul_5:z:0$lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_2AddV2lstm_cell_8/BiasAdd_1:output:0lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_9Mullstm_cell_8/add_2:z:0lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_3AddV2lstm_cell_8/Mul_9:z:0lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_1/MinimumMinimumlstm_cell_8/Add_3:z:0.lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_1Maximum'lstm_cell_8/clip_by_value_1/Minimum:z:0&lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????
lstm_cell_8/mul_10Mullstm_cell_8/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_2ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_2StridedSlice$lstm_cell_8/ReadVariableOp_2:value:0*lstm_cell_8/strided_slice_2/stack:output:0,lstm_cell_8/strided_slice_2/stack_1:output:0,lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_6MatMullstm_cell_8/mul_6:z:0$lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_4AddV2lstm_cell_8/BiasAdd_2:output:0lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/TanhTanhlstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_11Mullstm_cell_8/clip_by_value:z:0lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:??????????}
lstm_cell_8/add_5AddV2lstm_cell_8/mul_10:z:0lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/ReadVariableOp_3ReadVariableOp#lstm_cell_8_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_8/strided_slice_3StridedSlice$lstm_cell_8/ReadVariableOp_3:value:0*lstm_cell_8/strided_slice_3/stack:output:0,lstm_cell_8/strided_slice_3/stack_1:output:0,lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_8/MatMul_7MatMullstm_cell_8/mul_7:z:0$lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/add_6AddV2lstm_cell_8/BiasAdd_3:output:0lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_8/Mul_12Mullstm_cell_8/add_6:z:0lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/Add_7AddV2lstm_cell_8/Mul_12:z:0lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_8/clip_by_value_2/MinimumMinimumlstm_cell_8/Add_7:z:0.lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_8/clip_by_value_2Maximum'lstm_cell_8/clip_by_value_2/Minimum:z:0&lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_8/Tanh_1Tanhlstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_8/mul_13Mullstm_cell_8/clip_by_value_2:z:0lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_8_split_readvariableop_resource+lstm_cell_8_split_1_readvariableop_resource#lstm_cell_8_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_76818*
condR
while_cond_76817*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:???????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:???????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_8/ReadVariableOp^lstm_cell_8/ReadVariableOp_1^lstm_cell_8/ReadVariableOp_2^lstm_cell_8/ReadVariableOp_3!^lstm_cell_8/split/ReadVariableOp#^lstm_cell_8/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 28
lstm_cell_8/ReadVariableOplstm_cell_8/ReadVariableOp2<
lstm_cell_8/ReadVariableOp_1lstm_cell_8/ReadVariableOp_12<
lstm_cell_8/ReadVariableOp_2lstm_cell_8/ReadVariableOp_22<
lstm_cell_8/ReadVariableOp_3lstm_cell_8/ReadVariableOp_32D
 lstm_cell_8/split/ReadVariableOp lstm_cell_8/split/ReadVariableOp2H
"lstm_cell_8/split_1/ReadVariableOp"lstm_cell_8/split_1/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
k
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75897

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????o
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_75076

inputs
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74945o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73911

inputs

states
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??=[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2裬]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??^]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??u
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??`]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????t
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????p
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:??????????X
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????\
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:??????????\
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?X
Mul_8Muladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????^
Add_1AddV2	Mul_8:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_9Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_9:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????_
mul_10Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????]
mul_11Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????Y
add_5AddV2
mul_10:z:0
mul_11:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
Mul_12Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????_
Add_7AddV2
Mul_12:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????a
mul_13Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????\

Identity_1Identity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?=
?
__inference__traced_save_77820
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_lstm_lstm_cell_8_kernel_read_readvariableop@
<savev2_lstm_lstm_cell_8_recurrent_kernel_read_readvariableop4
0savev2_lstm_lstm_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop=
9savev2_adam_lstm_lstm_cell_8_kernel_m_read_readvariableopG
Csavev2_adam_lstm_lstm_cell_8_recurrent_kernel_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_8_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop=
9savev2_adam_lstm_lstm_cell_8_kernel_v_read_readvariableopG
Csavev2_adam_lstm_lstm_cell_8_recurrent_kernel_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_lstm_lstm_cell_8_kernel_read_readvariableop<savev2_lstm_lstm_cell_8_recurrent_kernel_read_readvariableop0savev2_lstm_lstm_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop9savev2_adam_lstm_lstm_cell_8_kernel_m_read_readvariableopCsavev2_adam_lstm_lstm_cell_8_recurrent_kernel_m_read_readvariableop7savev2_adam_lstm_lstm_cell_8_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop9savev2_adam_lstm_lstm_cell_8_kernel_v_read_readvariableopCsavev2_adam_lstm_lstm_cell_8_recurrent_kernel_v_read_readvariableop7savev2_adam_lstm_lstm_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :???:	?:: : : : : :
??:
??:?: : : : :???:	?::
??:
??:?:???:	?::
??:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:???:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&	"
 
_output_shapes
:
??:&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
:???:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:'#
!
_output_shapes
:???:%!

_output_shapes
:	?: 

_output_shapes
::&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?
?
*__inference_sequential_layer_call_fn_75059

inputs
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_74399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_lstm_layer_call_fn_75946
inputs_0
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74046p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?

?
@__inference_dense_layer_call_and_return_conditional_losses_74392

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?m
?
!__inference__traced_restore_77911
file_prefix:
%assignvariableop_embedding_embeddings:???2
assignvariableop_1_dense_kernel:	?+
assignvariableop_2_dense_bias:&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: >
*assignvariableop_8_lstm_lstm_cell_8_kernel:
??H
4assignvariableop_9_lstm_lstm_cell_8_recurrent_kernel:
??8
)assignvariableop_10_lstm_lstm_cell_8_bias:	?#
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: D
/assignvariableop_15_adam_embedding_embeddings_m:???:
'assignvariableop_16_adam_dense_kernel_m:	?3
%assignvariableop_17_adam_dense_bias_m:F
2assignvariableop_18_adam_lstm_lstm_cell_8_kernel_m:
??P
<assignvariableop_19_adam_lstm_lstm_cell_8_recurrent_kernel_m:
???
0assignvariableop_20_adam_lstm_lstm_cell_8_bias_m:	?D
/assignvariableop_21_adam_embedding_embeddings_v:???:
'assignvariableop_22_adam_dense_kernel_v:	?3
%assignvariableop_23_adam_dense_bias_v:F
2assignvariableop_24_adam_lstm_lstm_cell_8_kernel_v:
??P
<assignvariableop_25_adam_lstm_lstm_cell_8_recurrent_kernel_v:
???
0assignvariableop_26_adam_lstm_lstm_cell_8_bias_v:	?
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_lstm_lstm_cell_8_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_lstm_lstm_cell_8_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_lstm_lstm_cell_8_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_adam_embedding_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_dense_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_lstm_lstm_cell_8_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp<assignvariableop_19_adam_lstm_lstm_cell_8_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_lstm_lstm_cell_8_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_adam_embedding_embeddings_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp%assignvariableop_23_adam_dense_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_lstm_lstm_cell_8_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp<assignvariableop_25_adam_lstm_lstm_cell_8_recurrent_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_lstm_lstm_cell_8_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?	
while_body_74639
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_8_split_readvariableop_resource_0:
??B
3while_lstm_cell_8_split_1_readvariableop_resource_0:	??
+while_lstm_cell_8_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_8_split_readvariableop_resource:
??@
1while_lstm_cell_8_split_1_readvariableop_resource:	?=
)while_lstm_cell_8_readvariableop_resource:
???? while/lstm_cell_8/ReadVariableOp?"while/lstm_cell_8/ReadVariableOp_1?"while/lstm_cell_8/ReadVariableOp_2?"while/lstm_cell_8/ReadVariableOp_3?&while/lstm_cell_8/split/ReadVariableOp?(while/lstm_cell_8/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
!while/lstm_cell_8/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_likeFill*while/lstm_cell_8/ones_like/Shape:output:0*while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????d
while/lstm_cell_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout/MulMul$while/lstm_cell_8/ones_like:output:0(while/lstm_cell_8/dropout/Const:output:0*
T0*(
_output_shapes
:??????????s
while/lstm_cell_8/dropout/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_8/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_8/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ɩ?m
(while/lstm_cell_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_8/dropout/GreaterEqualGreaterEqual?while/lstm_cell_8/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/CastCast*while/lstm_cell_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
while/lstm_cell_8/dropout/Mul_1Mul!while/lstm_cell_8/dropout/Mul:z:0"while/lstm_cell_8/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_1/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_1/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2o
*while/lstm_cell_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_1/CastCast,while/lstm_cell_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_1/Mul_1Mul#while/lstm_cell_8/dropout_1/Mul:z:0$while/lstm_cell_8/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_2/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_2/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_8/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_2/CastCast,while/lstm_cell_8/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_2/Mul_1Mul#while/lstm_cell_8/dropout_2/Mul:z:0$while/lstm_cell_8/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_3/MulMul$while/lstm_cell_8/ones_like:output:0*while/lstm_cell_8/dropout_3/Const:output:0*
T0*(
_output_shapes
:??????????u
!while/lstm_cell_8/dropout_3/ShapeShape$while/lstm_cell_8/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_3/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??lo
*while/lstm_cell_8/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_3/CastCast,while/lstm_cell_8/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_3/Mul_1Mul#while/lstm_cell_8/dropout_3/Mul:z:0$while/lstm_cell_8/dropout_3/Cast:y:0*
T0*(
_output_shapes
:??????????f
#while/lstm_cell_8/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/ones_like_1Fill,while/lstm_cell_8/ones_like_1/Shape:output:0,while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_4/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_4/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_4/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_4/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??\o
*while/lstm_cell_8/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_4/CastCast,while/lstm_cell_8/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_4/Mul_1Mul#while/lstm_cell_8/dropout_4/Mul:z:0$while/lstm_cell_8/dropout_4/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_5/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_5/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_5/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_5/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2̥?o
*while/lstm_cell_8/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_5/CastCast,while/lstm_cell_8/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_5/Mul_1Mul#while/lstm_cell_8/dropout_5/Mul:z:0$while/lstm_cell_8/dropout_5/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_6/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_6/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_6/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_6/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??Do
*while/lstm_cell_8/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_6/CastCast,while/lstm_cell_8/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_6/Mul_1Mul#while/lstm_cell_8/dropout_6/Mul:z:0$while/lstm_cell_8/dropout_6/Cast:y:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_8/dropout_7/MulMul&while/lstm_cell_8/ones_like_1:output:0*while/lstm_cell_8/dropout_7/Const:output:0*
T0*(
_output_shapes
:??????????w
!while/lstm_cell_8/dropout_7/ShapeShape&while/lstm_cell_8/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_8/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_8/dropout_7/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??wo
*while/lstm_cell_8/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_8/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_8/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_8/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/dropout_7/CastCast,while/lstm_cell_8/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
!while/lstm_cell_8/dropout_7/Mul_1Mul#while/lstm_cell_8/dropout_7/Mul:z:0$while/lstm_cell_8/dropout_7/Cast:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_8/dropout/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_8/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:??????????c
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_8/split/ReadVariableOpReadVariableOp1while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0.while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_8/MatMulMatMulwhile/lstm_cell_8/mul:z:0 while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_1MatMulwhile/lstm_cell_8/mul_1:z:0 while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_2MatMulwhile/lstm_cell_8/mul_2:z:0 while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/MatMul_3MatMulwhile/lstm_cell_8/mul_3:z:0 while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_8/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_8/split_1Split,while/lstm_cell_8/split_1/split_dim:output:00while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_8/BiasAddBiasAdd"while/lstm_cell_8/MatMul:product:0"while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_1BiasAdd$while/lstm_cell_8/MatMul_1:product:0"while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_2BiasAdd$while/lstm_cell_8/MatMul_2:product:0"while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/BiasAdd_3BiasAdd$while/lstm_cell_8/MatMul_3:product:0"while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_4Mulwhile_placeholder_2%while/lstm_cell_8/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_5Mulwhile_placeholder_2%while/lstm_cell_8/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_6Mulwhile_placeholder_2%while/lstm_cell_8/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_7Mulwhile_placeholder_2%while/lstm_cell_8/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:???????????
 while/lstm_cell_8/ReadVariableOpReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_8/strided_sliceStridedSlice(while/lstm_cell_8/ReadVariableOp:value:0.while/lstm_cell_8/strided_slice/stack:output:00while/lstm_cell_8/strided_slice/stack_1:output:00while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_4MatMulwhile/lstm_cell_8/mul_4:z:0(while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/addAddV2"while/lstm_cell_8/BiasAdd:output:0$while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_8Mulwhile/lstm_cell_8/add:z:0 while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_1AddV2while/lstm_cell_8/Mul_8:z:0"while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_8/clip_by_value/MinimumMinimumwhile/lstm_cell_8/Add_1:z:02while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_8/clip_by_valueMaximum+while/lstm_cell_8/clip_by_value/Minimum:z:0*while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_1ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_1StridedSlice*while/lstm_cell_8/ReadVariableOp_1:value:00while/lstm_cell_8/strided_slice_1/stack:output:02while/lstm_cell_8/strided_slice_1/stack_1:output:02while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_5MatMulwhile/lstm_cell_8/mul_5:z:0*while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_2AddV2$while/lstm_cell_8/BiasAdd_1:output:0$while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_9Mulwhile/lstm_cell_8/add_2:z:0"while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_3AddV2while/lstm_cell_8/Mul_9:z:0"while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_1/MinimumMinimumwhile/lstm_cell_8/Add_3:z:04while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_1Maximum-while/lstm_cell_8/clip_by_value_1/Minimum:z:0,while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_10Mul%while/lstm_cell_8/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_2ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_2StridedSlice*while/lstm_cell_8/ReadVariableOp_2:value:00while/lstm_cell_8/strided_slice_2/stack:output:02while/lstm_cell_8/strided_slice_2/stack_1:output:02while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_6MatMulwhile/lstm_cell_8/mul_6:z:0*while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_4AddV2$while/lstm_cell_8/BiasAdd_2:output:0$while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_8/TanhTanhwhile/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_11Mul#while/lstm_cell_8/clip_by_value:z:0while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_5AddV2while/lstm_cell_8/mul_10:z:0while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_8/ReadVariableOp_3ReadVariableOp+while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_8/strided_slice_3StridedSlice*while/lstm_cell_8/ReadVariableOp_3:value:00while/lstm_cell_8/strided_slice_3/stack:output:02while/lstm_cell_8/strided_slice_3/stack_1:output:02while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_8/MatMul_7MatMulwhile/lstm_cell_8/mul_7:z:0*while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/add_6AddV2$while/lstm_cell_8/BiasAdd_3:output:0$while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_8/Mul_12Mulwhile/lstm_cell_8/add_6:z:0"while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/Add_7AddV2while/lstm_cell_8/Mul_12:z:0"while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_8/clip_by_value_2/MinimumMinimumwhile/lstm_cell_8/Add_7:z:04while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_8/clip_by_value_2Maximum-while/lstm_cell_8/clip_by_value_2/Minimum:z:0,while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_8/Tanh_1Tanhwhile/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_8/mul_13Mul%while/lstm_cell_8/clip_by_value_2:z:0while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_8/mul_13:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_8/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_8/ReadVariableOp#^while/lstm_cell_8/ReadVariableOp_1#^while/lstm_cell_8/ReadVariableOp_2#^while/lstm_cell_8/ReadVariableOp_3'^while/lstm_cell_8/split/ReadVariableOp)^while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_8_readvariableop_resource+while_lstm_cell_8_readvariableop_resource_0"h
1while_lstm_cell_8_split_1_readvariableop_resource3while_lstm_cell_8_split_1_readvariableop_resource_0"d
/while_lstm_cell_8_split_readvariableop_resource1while_lstm_cell_8_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_8/ReadVariableOp while/lstm_cell_8/ReadVariableOp2H
"while/lstm_cell_8/ReadVariableOp_1"while/lstm_cell_8/ReadVariableOp_12H
"while/lstm_cell_8/ReadVariableOp_2"while/lstm_cell_8/ReadVariableOp_22H
"while/lstm_cell_8/ReadVariableOp_3"while/lstm_cell_8/ReadVariableOp_32P
&while/lstm_cell_8/split/ReadVariableOp&while/lstm_cell_8/split/ReadVariableOp2T
(while/lstm_cell_8/split_1/ReadVariableOp(while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?U
?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73631

inputs

states
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??~
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:??????????Y
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????[
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:??????????`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:??????????`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:??????????`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????]
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????]
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????]
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????]
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?X
Mul_8Muladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????^
Add_1AddV2	Mul_8:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_9Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_9:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????_
mul_10Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????]
mul_11Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????Y
add_5AddV2
mul_10:z:0
mul_11:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?]
Mul_12Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????_
Add_7AddV2
Mul_12:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????a
mul_13Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????\

Identity_1Identity
mul_13:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
#__inference_signature_wrapper_75042
embedding_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:	?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_73446o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:??????????
)
_user_specified_nameembedding_input
?
j
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74080

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
ٳ
?
 sequential_lstm_while_body_73285<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0U
Asequential_lstm_while_lstm_cell_8_split_readvariableop_resource_0:
??R
Csequential_lstm_while_lstm_cell_8_split_1_readvariableop_resource_0:	?O
;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0:
??"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorS
?sequential_lstm_while_lstm_cell_8_split_readvariableop_resource:
??P
Asequential_lstm_while_lstm_cell_8_split_1_readvariableop_resource:	?M
9sequential_lstm_while_lstm_cell_8_readvariableop_resource:
????0sequential/lstm/while/lstm_cell_8/ReadVariableOp?2sequential/lstm/while/lstm_cell_8/ReadVariableOp_1?2sequential/lstm/while/lstm_cell_8/ReadVariableOp_2?2sequential/lstm/while/lstm_cell_8/ReadVariableOp_3?6sequential/lstm/while/lstm_cell_8/split/ReadVariableOp?8sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOp?
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
1sequential/lstm/while/lstm_cell_8/ones_like/ShapeShape@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:v
1sequential/lstm/while/lstm_cell_8/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
+sequential/lstm/while/lstm_cell_8/ones_likeFill:sequential/lstm/while/lstm_cell_8/ones_like/Shape:output:0:sequential/lstm/while/lstm_cell_8/ones_like/Const:output:0*
T0*(
_output_shapes
:???????????
3sequential/lstm/while/lstm_cell_8/ones_like_1/ShapeShape#sequential_lstm_while_placeholder_2*
T0*
_output_shapes
:x
3sequential/lstm/while/lstm_cell_8/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-sequential/lstm/while/lstm_cell_8/ones_like_1Fill<sequential/lstm/while/lstm_cell_8/ones_like_1/Shape:output:0<sequential/lstm/while/lstm_cell_8/ones_like_1/Const:output:0*
T0*(
_output_shapes
:???????????
%sequential/lstm/while/lstm_cell_8/mulMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_1Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_2Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_3Mul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/lstm/while/lstm_cell_8/ones_like:output:0*
T0*(
_output_shapes
:??????????s
1sequential/lstm/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
6sequential/lstm/while/lstm_cell_8/split/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_8_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
'sequential/lstm/while/lstm_cell_8/splitSplit:sequential/lstm/while/lstm_cell_8/split/split_dim:output:0>sequential/lstm/while/lstm_cell_8/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
(sequential/lstm/while/lstm_cell_8/MatMulMatMul)sequential/lstm/while/lstm_cell_8/mul:z:00sequential/lstm/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:???????????
*sequential/lstm/while/lstm_cell_8/MatMul_1MatMul+sequential/lstm/while/lstm_cell_8/mul_1:z:00sequential/lstm/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:???????????
*sequential/lstm/while/lstm_cell_8/MatMul_2MatMul+sequential/lstm/while/lstm_cell_8/mul_2:z:00sequential/lstm/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:???????????
*sequential/lstm/while/lstm_cell_8/MatMul_3MatMul+sequential/lstm/while/lstm_cell_8/mul_3:z:00sequential/lstm/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:??????????u
3sequential/lstm/while/lstm_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOpReadVariableOpCsequential_lstm_while_lstm_cell_8_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
)sequential/lstm/while/lstm_cell_8/split_1Split<sequential/lstm/while/lstm_cell_8/split_1/split_dim:output:0@sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
)sequential/lstm/while/lstm_cell_8/BiasAddBiasAdd2sequential/lstm/while/lstm_cell_8/MatMul:product:02sequential/lstm/while/lstm_cell_8/split_1:output:0*
T0*(
_output_shapes
:???????????
+sequential/lstm/while/lstm_cell_8/BiasAdd_1BiasAdd4sequential/lstm/while/lstm_cell_8/MatMul_1:product:02sequential/lstm/while/lstm_cell_8/split_1:output:1*
T0*(
_output_shapes
:???????????
+sequential/lstm/while/lstm_cell_8/BiasAdd_2BiasAdd4sequential/lstm/while/lstm_cell_8/MatMul_2:product:02sequential/lstm/while/lstm_cell_8/split_1:output:2*
T0*(
_output_shapes
:???????????
+sequential/lstm/while/lstm_cell_8/BiasAdd_3BiasAdd4sequential/lstm/while/lstm_cell_8/MatMul_3:product:02sequential/lstm/while/lstm_cell_8/split_1:output:3*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_4Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_5Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_6Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/mul_7Mul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell_8/ones_like_1:output:0*
T0*(
_output_shapes
:???????????
0sequential/lstm/while/lstm_cell_8/ReadVariableOpReadVariableOp;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
5sequential/lstm/while/lstm_cell_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential/lstm/while/lstm_cell_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
7sequential/lstm/while/lstm_cell_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential/lstm/while/lstm_cell_8/strided_sliceStridedSlice8sequential/lstm/while/lstm_cell_8/ReadVariableOp:value:0>sequential/lstm/while/lstm_cell_8/strided_slice/stack:output:0@sequential/lstm/while/lstm_cell_8/strided_slice/stack_1:output:0@sequential/lstm/while/lstm_cell_8/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
*sequential/lstm/while/lstm_cell_8/MatMul_4MatMul+sequential/lstm/while/lstm_cell_8/mul_4:z:08sequential/lstm/while/lstm_cell_8/strided_slice:output:0*
T0*(
_output_shapes
:???????????
%sequential/lstm/while/lstm_cell_8/addAddV22sequential/lstm/while/lstm_cell_8/BiasAdd:output:04sequential/lstm/while/lstm_cell_8/MatMul_4:product:0*
T0*(
_output_shapes
:??????????l
'sequential/lstm/while/lstm_cell_8/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>n
)sequential/lstm/while/lstm_cell_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
'sequential/lstm/while/lstm_cell_8/Mul_8Mul)sequential/lstm/while/lstm_cell_8/add:z:00sequential/lstm/while/lstm_cell_8/Const:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/Add_1AddV2+sequential/lstm/while/lstm_cell_8/Mul_8:z:02sequential/lstm/while/lstm_cell_8/Const_1:output:0*
T0*(
_output_shapes
:??????????~
9sequential/lstm/while/lstm_cell_8/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential/lstm/while/lstm_cell_8/clip_by_value/MinimumMinimum+sequential/lstm/while/lstm_cell_8/Add_1:z:0Bsequential/lstm/while/lstm_cell_8/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????v
1sequential/lstm/while/lstm_cell_8/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/sequential/lstm/while/lstm_cell_8/clip_by_valueMaximum;sequential/lstm/while/lstm_cell_8/clip_by_value/Minimum:z:0:sequential/lstm/while/lstm_cell_8/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_1ReadVariableOp;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
7sequential/lstm/while/lstm_cell_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
9sequential/lstm/while/lstm_cell_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
9sequential/lstm/while/lstm_cell_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
1sequential/lstm/while/lstm_cell_8/strided_slice_1StridedSlice:sequential/lstm/while/lstm_cell_8/ReadVariableOp_1:value:0@sequential/lstm/while/lstm_cell_8/strided_slice_1/stack:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_1/stack_1:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
*sequential/lstm/while/lstm_cell_8/MatMul_5MatMul+sequential/lstm/while/lstm_cell_8/mul_5:z:0:sequential/lstm/while/lstm_cell_8/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/add_2AddV24sequential/lstm/while/lstm_cell_8/BiasAdd_1:output:04sequential/lstm/while/lstm_cell_8/MatMul_5:product:0*
T0*(
_output_shapes
:??????????n
)sequential/lstm/while/lstm_cell_8/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>n
)sequential/lstm/while/lstm_cell_8/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
'sequential/lstm/while/lstm_cell_8/Mul_9Mul+sequential/lstm/while/lstm_cell_8/add_2:z:02sequential/lstm/while/lstm_cell_8/Const_2:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/Add_3AddV2+sequential/lstm/while/lstm_cell_8/Mul_9:z:02sequential/lstm/while/lstm_cell_8/Const_3:output:0*
T0*(
_output_shapes
:???????????
;sequential/lstm/while/lstm_cell_8/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
9sequential/lstm/while/lstm_cell_8/clip_by_value_1/MinimumMinimum+sequential/lstm/while/lstm_cell_8/Add_3:z:0Dsequential/lstm/while/lstm_cell_8/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????x
3sequential/lstm/while/lstm_cell_8/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1sequential/lstm/while/lstm_cell_8/clip_by_value_1Maximum=sequential/lstm/while/lstm_cell_8/clip_by_value_1/Minimum:z:0<sequential/lstm/while/lstm_cell_8/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
(sequential/lstm/while/lstm_cell_8/mul_10Mul5sequential/lstm/while/lstm_cell_8/clip_by_value_1:z:0#sequential_lstm_while_placeholder_3*
T0*(
_output_shapes
:???????????
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_2ReadVariableOp;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
7sequential/lstm/while/lstm_cell_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
9sequential/lstm/while/lstm_cell_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
9sequential/lstm/while/lstm_cell_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
1sequential/lstm/while/lstm_cell_8/strided_slice_2StridedSlice:sequential/lstm/while/lstm_cell_8/ReadVariableOp_2:value:0@sequential/lstm/while/lstm_cell_8/strided_slice_2/stack:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_2/stack_1:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
*sequential/lstm/while/lstm_cell_8/MatMul_6MatMul+sequential/lstm/while/lstm_cell_8/mul_6:z:0:sequential/lstm/while/lstm_cell_8/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/add_4AddV24sequential/lstm/while/lstm_cell_8/BiasAdd_2:output:04sequential/lstm/while/lstm_cell_8/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
&sequential/lstm/while/lstm_cell_8/TanhTanh+sequential/lstm/while/lstm_cell_8/add_4:z:0*
T0*(
_output_shapes
:???????????
(sequential/lstm/while/lstm_cell_8/mul_11Mul3sequential/lstm/while/lstm_cell_8/clip_by_value:z:0*sequential/lstm/while/lstm_cell_8/Tanh:y:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/add_5AddV2,sequential/lstm/while/lstm_cell_8/mul_10:z:0,sequential/lstm/while/lstm_cell_8/mul_11:z:0*
T0*(
_output_shapes
:???????????
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_3ReadVariableOp;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
7sequential/lstm/while/lstm_cell_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
9sequential/lstm/while/lstm_cell_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
9sequential/lstm/while/lstm_cell_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
1sequential/lstm/while/lstm_cell_8/strided_slice_3StridedSlice:sequential/lstm/while/lstm_cell_8/ReadVariableOp_3:value:0@sequential/lstm/while/lstm_cell_8/strided_slice_3/stack:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_3/stack_1:output:0Bsequential/lstm/while/lstm_cell_8/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
*sequential/lstm/while/lstm_cell_8/MatMul_7MatMul+sequential/lstm/while/lstm_cell_8/mul_7:z:0:sequential/lstm/while/lstm_cell_8/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/add_6AddV24sequential/lstm/while/lstm_cell_8/BiasAdd_3:output:04sequential/lstm/while/lstm_cell_8/MatMul_7:product:0*
T0*(
_output_shapes
:??????????n
)sequential/lstm/while/lstm_cell_8/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>n
)sequential/lstm/while/lstm_cell_8/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
(sequential/lstm/while/lstm_cell_8/Mul_12Mul+sequential/lstm/while/lstm_cell_8/add_6:z:02sequential/lstm/while/lstm_cell_8/Const_4:output:0*
T0*(
_output_shapes
:???????????
'sequential/lstm/while/lstm_cell_8/Add_7AddV2,sequential/lstm/while/lstm_cell_8/Mul_12:z:02sequential/lstm/while/lstm_cell_8/Const_5:output:0*
T0*(
_output_shapes
:???????????
;sequential/lstm/while/lstm_cell_8/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
9sequential/lstm/while/lstm_cell_8/clip_by_value_2/MinimumMinimum+sequential/lstm/while/lstm_cell_8/Add_7:z:0Dsequential/lstm/while/lstm_cell_8/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????x
3sequential/lstm/while/lstm_cell_8/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1sequential/lstm/while/lstm_cell_8/clip_by_value_2Maximum=sequential/lstm/while/lstm_cell_8/clip_by_value_2/Minimum:z:0<sequential/lstm/while/lstm_cell_8/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
(sequential/lstm/while/lstm_cell_8/Tanh_1Tanh+sequential/lstm/while/lstm_cell_8/add_5:z:0*
T0*(
_output_shapes
:???????????
(sequential/lstm/while/lstm_cell_8/mul_13Mul5sequential/lstm/while/lstm_cell_8/clip_by_value_2:z:0,sequential/lstm/while/lstm_cell_8/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder,sequential/lstm/while/lstm_cell_8/mul_13:z:0*
_output_shapes
: *
element_dtype0:???]
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: ?
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ?
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ?
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ?
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ?
 sequential/lstm/while/Identity_4Identity,sequential/lstm/while/lstm_cell_8/mul_13:z:0^sequential/lstm/while/NoOp*
T0*(
_output_shapes
:???????????
 sequential/lstm/while/Identity_5Identity+sequential/lstm/while/lstm_cell_8/add_5:z:0^sequential/lstm/while/NoOp*
T0*(
_output_shapes
:???????????
sequential/lstm/while/NoOpNoOp1^sequential/lstm/while/lstm_cell_8/ReadVariableOp3^sequential/lstm/while/lstm_cell_8/ReadVariableOp_13^sequential/lstm/while/lstm_cell_8/ReadVariableOp_23^sequential/lstm/while/lstm_cell_8/ReadVariableOp_37^sequential/lstm/while/lstm_cell_8/split/ReadVariableOp9^sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"x
9sequential_lstm_while_lstm_cell_8_readvariableop_resource;sequential_lstm_while_lstm_cell_8_readvariableop_resource_0"?
Asequential_lstm_while_lstm_cell_8_split_1_readvariableop_resourceCsequential_lstm_while_lstm_cell_8_split_1_readvariableop_resource_0"?
?sequential_lstm_while_lstm_cell_8_split_readvariableop_resourceAsequential_lstm_while_lstm_cell_8_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"?
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2d
0sequential/lstm/while/lstm_cell_8/ReadVariableOp0sequential/lstm/while/lstm_cell_8/ReadVariableOp2h
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_12sequential/lstm/while/lstm_cell_8/ReadVariableOp_12h
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_22sequential/lstm/while/lstm_cell_8/ReadVariableOp_22h
2sequential/lstm/while/lstm_cell_8/ReadVariableOp_32sequential/lstm/while/lstm_cell_8/ReadVariableOp_32p
6sequential/lstm/while/lstm_cell_8/split/ReadVariableOp6sequential/lstm/while/lstm_cell_8/split/ReadVariableOp2t
8sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOp8sequential/lstm/while/lstm_cell_8/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_73977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_73977___redundant_placeholder03
/while_while_cond_73977___redundant_placeholder13
/while_while_cond_73977___redundant_placeholder23
/while_while_cond_73977___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
j
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_73455

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????q

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?"
?
while_body_73978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_8_74002_0:
??(
while_lstm_cell_8_74004_0:	?-
while_lstm_cell_8_74006_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_8_74002:
??&
while_lstm_cell_8_74004:	?+
while_lstm_cell_8_74006:
????)while/lstm_cell_8/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_74002_0while_lstm_cell_8_74004_0while_lstm_cell_8_74006_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_73911?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_8_74002while_lstm_cell_8_74002_0"4
while_lstm_cell_8_74004while_lstm_cell_8_74004_0"4
while_lstm_cell_8_74006while_lstm_cell_8_74006_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_76817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_76817___redundant_placeholder03
/while_while_cond_76817___redundant_placeholder13
/while_while_cond_76817___redundant_placeholder23
/while_while_cond_76817___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
M
1__inference_spatial_dropout1d_layer_call_fn_75865

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_74080f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_lstm_layer_call_fn_75957

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_74373p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
L
embedding_input9
!serving_default_embedding_input:0??????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
c__call__
*d&call_and_return_all_conditional_losses
e_default_save_signature"
_tf_keras_sequential
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratemWmXmY%mZ&m['m\v]v^v_%v`&va'vb"
	optimizer
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
c__call__
e_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
,
nserving_default"
signature_map
):'???2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
?
7
state_size

%kernel
&recurrent_kernel
'bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

<states
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)
??2lstm/lstm_cell_8/kernel
5:3
??2!lstm/lstm_cell_8/recurrent_kernel
$:"?2lstm/lstm_cell_8/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
8	variables
9trainable_variables
:regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ntotal
	Ocount
P	variables
Q	keras_api"
_tf_keras_metric
^
	Rtotal
	Scount
T
_fn_kwargs
U	variables
V	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
N0
O1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
R0
S1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
.:,???2Adam/embedding/embeddings/m
$:"	?2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.
??2Adam/lstm/lstm_cell_8/kernel/m
::8
??2(Adam/lstm/lstm_cell_8/recurrent_kernel/m
):'?2Adam/lstm/lstm_cell_8/bias/m
.:,???2Adam/embedding/embeddings/v
$:"	?2Adam/dense/kernel/v
:2Adam/dense/bias/v
0:.
??2Adam/lstm/lstm_cell_8/kernel/v
::8
??2(Adam/lstm/lstm_cell_8/recurrent_kernel/v
):'?2Adam/lstm/lstm_cell_8/bias/v
?2?
*__inference_sequential_layer_call_fn_74414
*__inference_sequential_layer_call_fn_75059
*__inference_sequential_layer_call_fn_75076
*__inference_sequential_layer_call_fn_74977?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_75382
E__inference_sequential_layer_call_and_return_conditional_losses_75833
E__inference_sequential_layer_call_and_return_conditional_losses_74997
E__inference_sequential_layer_call_and_return_conditional_losses_75017?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_73446embedding_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_embedding_layer_call_fn_75840?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_75850?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_spatial_dropout1d_layer_call_fn_75855
1__inference_spatial_dropout1d_layer_call_fn_75860
1__inference_spatial_dropout1d_layer_call_fn_75865
1__inference_spatial_dropout1d_layer_call_fn_75870?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75875
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75897
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75902
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75924?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_lstm_layer_call_fn_75935
$__inference_lstm_layer_call_fn_75946
$__inference_lstm_layer_call_fn_75957
$__inference_lstm_layer_call_fn_75968?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_lstm_layer_call_and_return_conditional_losses_76260
?__inference_lstm_layer_call_and_return_conditional_losses_76680
?__inference_lstm_layer_call_and_return_conditional_losses_76972
?__inference_lstm_layer_call_and_return_conditional_losses_77392?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_77401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_77412?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_75042embedding_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_lstm_cell_8_layer_call_fn_77429
+__inference_lstm_cell_8_layer_call_fn_77446?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77549
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77716?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_73446r%'&9?6
/?,
*?'
embedding_input??????????
? "-?*
(
dense?
dense??????????
@__inference_dense_layer_call_and_return_conditional_losses_77412]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_77401P0?-
&?#
!?
inputs??????????
? "???????????
D__inference_embedding_layer_call_and_return_conditional_losses_75850b0?-
&?#
!?
inputs??????????
? "+?(
!?
0???????????
? ?
)__inference_embedding_layer_call_fn_75840U0?-
&?#
!?
inputs??????????
? "?????????????
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77549?%'&???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_8_layer_call_and_return_conditional_losses_77716?%'&???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_8_layer_call_fn_77429?%'&???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_8_layer_call_fn_77446?%'&???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
?__inference_lstm_layer_call_and_return_conditional_losses_76260%'&P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#
?
0??????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_76680%'&P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#
?
0??????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_76972p%'&A?>
7?4
&?#
inputs???????????

 
p 

 
? "&?#
?
0??????????
? ?
?__inference_lstm_layer_call_and_return_conditional_losses_77392p%'&A?>
7?4
&?#
inputs???????????

 
p

 
? "&?#
?
0??????????
? ?
$__inference_lstm_layer_call_fn_75935r%'&P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "????????????
$__inference_lstm_layer_call_fn_75946r%'&P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "????????????
$__inference_lstm_layer_call_fn_75957c%'&A?>
7?4
&?#
inputs???????????

 
p 

 
? "????????????
$__inference_lstm_layer_call_fn_75968c%'&A?>
7?4
&?#
inputs???????????

 
p

 
? "????????????
E__inference_sequential_layer_call_and_return_conditional_losses_74997r%'&A?>
7?4
*?'
embedding_input??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_75017r%'&A?>
7?4
*?'
embedding_input??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_75382i%'&8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_75833i%'&8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_74414e%'&A?>
7?4
*?'
embedding_input??????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_74977e%'&A?>
7?4
*?'
embedding_input??????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_75059\%'&8?5
.?+
!?
inputs??????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_75076\%'&8?5
.?+
!?
inputs??????????
p

 
? "???????????
#__inference_signature_wrapper_75042?%'&L?I
? 
B??
=
embedding_input*?'
embedding_input??????????"-?*
(
dense?
dense??????????
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75875?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75897?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75902h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
L__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_75924h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
1__inference_spatial_dropout1d_layer_call_fn_75855{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
1__inference_spatial_dropout1d_layer_call_fn_75860{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
1__inference_spatial_dropout1d_layer_call_fn_75865[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
1__inference_spatial_dropout1d_layer_call_fn_75870[9?6
/?,
&?#
inputs???????????
p
? "????????????