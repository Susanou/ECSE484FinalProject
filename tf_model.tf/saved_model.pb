??
?&?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
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
A
SelectV2
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.0-dev202204062v1.12.1-73640-g0c9c5a41a758??
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?N*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?N*
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
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
}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_65*
value_dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1413*
value_dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
޿
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NBetBdeBlaBleBlesBdesBunBquiBqueBdansB?BenBduBauBjeBcommeBsurBsonBvousBestBceBuneBtoutBneBseBpourBaBilBplusBsaBauxBo?BmonBpasBnousBsansBsesBcestBparBavecBtonBmaBdunBsousBtuBcesBcoeurBelleBleurBleursBsiBfaitByeuxBquandBcielBtaBmeBouBtousBtesBonBdontBsontBbienBoBvosBcetteBmaisB?meBmesBdieuBilsBnosBversBmoiBquilBluiBnuitBdeuxB?BvotreBjourBtoujoursBdouxBditBquonBtoiBteBainsiBrienBduneBgrandBquunBvieuxBsoleilBmortBjaiBlonBvoixBniBfondBmondeBjamaisBsuisBnotreBfrontBbrasBcarBvieBlombreBbeauBterreBm?meBsoirBtouteBgrandsBnoirBontBnestBventBdoncBcorpsBquelqueBmainBbelleBlhommeBlamourBt?teBboisBtempsBvaBfleursBsombreBtoutesBencorBfleurBamourBr?veBseulBl?BpeuBchaqueBlairBpiedsBfautBfemmeBvoirBpauvreBvoitBquelleBdorBl?meB?BfontBmerBbeaut?BpuisBloinBchosesB?taitBmoinsBcieuxBdevantBparmiBtraversBtristeBpleinBh?lasBfeuBmainsBquelBtombeBsenBaussiBboucheBluneBhommeBregardBtantBbasBnatureBsangBlautreBbruitB?BnonBvontBp?leBloeilBm?reBapr?sBahBpleursBenfantBpeutBbonBhautBbeauxBfaireBencoreBcheveuxBmortsBoeilBjeuneBnoirsBblancheBentreBpr?sBlumi?reBcoeursBveuxBtropBcetBmalB?treBvoiciBlitBbaiserBbaisersBvinBflammeBsortBautourBcelaBnaBdouleurBchairBleauBsoitBsaisBohBvoil?BjoieBgloireBtourBdireBmotBvientBveutBpourquoiBmilleBlongBfoisBchanteBailesBnoireBlheureBmatinByBpiti?BrireBpleureBtr?sBpens?eBbleuBritBespritBlespritBvoisBrayonBjoursBlenfantBseinBparisBimmenseBmotsBangeBquuneBcheminBpeurBnomBchoseBvuBsembleBprofondBpierreBpourtantBsouventBpiedBparfumsBrosesBtombeauBnulBt?n?bresBcharmantBquenBgrandeBgouffreBsaitBjoyeuxBfraisBpeineBpetitsBparfumBcalmeBflotsBprendBcherBfinBouiBlongsBceuxBnuitsBmilieuBquoiBprisBpetitBdouceBfilsBd?j?BdosBsouffleBlaisseBbonneBquilsBalorsBrayonsBpointBresteBboutBhommesBeuxB	cependantBoiseauxBautreBairBr?vesBroseBd?treBsouvenirBseraBp?reBiiBma?treBclairBiBpass?BavaitBparfoisBchlorisBpleinsBfemmesBcoupBsatanBavonsBfroidB	sylvandreBclart?BbonheurB?tesBsensBporteBchantB	rosalindeBfilleBmieuxBvagueBbordBpo?teBdisBpeupleBheureuxBvainBchansonBblancBavezB?trangeBplaisirBmyrtilBroiBfutBaileB	printempsBlorBvivreBenfantsBprofondeBmis?reBlazurBmyst?reBavoirBmarbreBansBaimeB	longtempsBd?sirBdouceurBblanchesBfaceBdamourBdautresBdoigtsBj?taisBc?taitBformeBcrisBqu?BlorsqueBlivreBbellesBvitBlunBsourireBfaisBarbresBremordsBp?lesBpurBmetBlongueBsilenceBderri?reBlherbeBjoueBregardeBlenferBiiiBfouleBavantBamoursBsoisBsoeurBrougeBpleineBprendsBpaixBdouteBseuleB?mesBneigeBdentsBmusiqueBgr?ceBdivinBpalaisBangesBsilBpeut?treBfolleBlieuBjaimeBfun?bresBvieilleBsecretBmorneBmourirBdo?BcroixBvertsBseinsBlartBdonBtroisBraceBchampsBpasseBmarcheBlarmesBpendantBamerBdepuisBgenouxBsommeilBnontBmonteBtientBtelBblancsBprendreBceluiBpartoutBjusqu?BfleuveBpaysBpasserBcoinBsommesBlargeBvraiBfortBjusquauBferBlhorizonBmaiBdonneBvaisBombreBvilleBch?reBautantBvraimentBsublimeBraisonBsinistreBlhiverBb?teBflancBdestinBreineBiciB
myst?rieuxBmauvaisBrobeBpetiteBgorgeBsentBsaintBpo?teBjeunesBforceBfaisaitBdouleursBvolBsongeBjardinBjetteBvoyantBvertBrouteBjadisBlaubeBivBfitBverBtrouverBlab?meBfroideBjalouxBtandisBpluieBj?susBayantBnenBaimerBsoleilsBgardeBplisBlangeBchacunBsolBenfinBsuperbeBsavoirBricheBprofondsBmomentBmisBl?vresBamoureuxBvisBmanteauBviergeB	doucementBorBallonsBluitBcadavreBoiseauBchercherBautresBloiseauB18BviensBquelquesBamantBvermeilBtrouveBdroitBcroitB?normeBpremierBd?sertBlueurBfoiBbrumeBmalgr?BbranchesBvaguesBpassantBbriseBfrontsBblondeBlesbosBbleusBm?leBspectreBsi?cleBsainteBchambreB
aujourdhuiBafinBdeuilBviennentBflotBdemainBvideBpuisqueBmiroirBlisBdormirBpauvresBjavaisBassezB	lentementBlourdsBd?sBtendreBcendreBventreBlampeBcontreBhumaineBfauveBcreuxBvenirBmonstreBfaiteBparadisBmystiqueBlasBdavoirB	vieillardB	t?n?breuxBtoileBscienceBparleBl?vreBvolupt?B	vainqueurBmaisonBaupr?sBvivantBseigneurBlourdBgrosBvasteBtombeauxBmaigreBfoyerBcriBheureBpartBhideuxBeauxBbutBdeboutBcouchantBespritsBentendBjeunesseBfatalBsestBtoursBlastreBamantsBsoupirsBpureBpersonneBl?basBv?nusBluniversB
maintenantBlinfiniBproieBn?antBfuitBfaimBvoileBlueursBjuinBhumbleBcouleursBchristBastresBvBpouvoirBbrilleBadieuBsageBmurBf?teBcommentB?toileBsombresBnusBterribleBmontBhaineBsoudainBpoisonBdieuxBloiBlinceulB	solitaireBtravailBpousseBhierBpenseBlhorreurBplaceBombresBdurBdoigtBvoilesBpainBjauneBfr?reBfaisantBcourtBchezBchevalB	charmantsBtoitBhumainBdevientBtrembleBmorteBcouleurB	cherchantBmersBbalBgraveBgo?tBextaseBhaleineBchercheBregardsBplaineBnuBl?veBtemp?teBsyBrefletBpareilBlarbreBcelleBvautBvastesBlointainBcouronneBallerBventsBcharmeBhorribleBvieilBtableBpuBgoutteBfroidsBdeauBchanterBcauseBrouleBsupr?meBsaintsBpaupi?reBnouveauBmeurtBdortBairsBlondeBfor?tBc?t?BbesoinB	splendeurBloubliBchienBfant?meBdazurBceuxl?BvieillesBfaitesBcroireBviteBsouffreBosBtortBpri?reBl?toileBfauxBfaroucheBboireBvaisseauBtroupeauBsuitBnaiBmasqueBespoirBavrilBarbreBcouBvivantsBrueBpencheBnyBchampBcentBvoyageBsoyezBseulsBparlerBpareilsBflammesBtoitsBpr?sentBorgueilBnezBlibreBlextaseBcreuseB?ternelBv?rit?BtasBrendBmenBchantsBpassionBdanseBroisBcoupsBbient?tB	semblableBm?moireBjuanBcrimeBvisageBpresqueBpoitrineBlennuiBsoirsBserpentBlespoirBjusteBobscurBdivineBl?cheBlionBd?sirsBlieuxBnidBmoisBlarmeBlaileBhasardBdiableBlespaceBpens?esBbouchesBpursBfousBdargentBbruitsBantiqueBseuilBquauBlauroreBfr?leBdernierBtr?sorBsuivreBsourdBprunelleBlorgueilBlargesBmurmureBnuagesBfor?tsB	ma?tresseBgrisBenti?reBaudessusBaffreuxB	tristesseBtempleBfermeBtomberBvasBt?tesBtrousBsolitudeBremplitB	firmamentBfaibleBdautomneBcroisBchatBcacheB	?ternelleB?taientBvintB	tremblantBoreilleBfranceBplut?tBplongerBmursBjenBfruitBe?tBdescendB	souvenirsBarmesBsorteBportesBouvertBnuageBfouBodeurBmontreBfant?mesBchevauxBnidsBnavireBdairainBcaresseBbijouxBvilBsonoreBouvrantBfaitsBeauBveilleBsoeursBseptBsemblaitBpensifB	l?ternit?BlourdeBinf?meBdombreBcouleBam?reBvolupt?sBquelquefoisBpeauB
m?lancolieBl?treBluim?meBfierBellesBchasteBbl?meB?toilesBpetitesBoeuvreBmonstresBinstantBgensBdenB	cimeti?reBplaisirsBmauditBrocherBmondesBl?ternelBfortsBbleueB
malheureuxBveuxtuBtelleBr?veurBinstantsBfillesBtableauxBmadameBforteBdabordBcesseBplafondBl?gersBcygneBbruneBallaitBviiBtardB	ruisseauxBouvreBnueBhauteBdameBcheminsB	prunellesBfosseBdivoireBdelleBcombienBvenuBsoupirBnombreBmordBcol?reBperlesBfumeBestceBd?monsBdoitBaimableBseraitBh?rosBfeuilleBsertBroueBpleurerBglobeBchantantBviBsurtoutBsouritBlangueBdorsBtournezBmalheurBcandeurB?tresBreposBradieuxBpo?tesBfeuxBbeaut?sB?pauleBvoyezBposeBparoleBflancsBverseBsauvageBhorizonBguerreBd?t?BclairsBsortirBrisBsoldatBr?vantBimmondeBg?nieBeffortsBbr?leBastreBviiiBsouciBparceBmonsieurBmarcherBdisaitBamiBvifBviendraBs?rBp?treBplainteBpardonBn?taitBmuetBlangueurBivreBd?brisBdroiteBchantaitBstatueBromeBrendreBpierresBmoim?meBhorreurBavaientB1843BtroubleBsereinBsemblentBpuitsBinconnusBobscureBm?chantBmoqueurBmetsBgr?cesBfuirBflambeauBb?tesB?tantBretourBlireBjoyeuseBga?t?BchangeBassisB	amoureuseBpalpiteBcr?atureBaitBvoyageurBtourneBsuivantBpeuventBheuresBformesBeffortBditesBcercueilBab?meBsixBquatreBpremi?reBcolBtr?neB	seulementBnobleB	autrefoisBsentirBloupsB	charmanteBtra?neB	squeletteBsonneBquelquunBprofondeursBmiroirsBgesteBgazonBth??treBsalutBr?leBmousseBlaissezBenvieBconnuBvoyaitBveloursBsecretsBrameauxBquantBperduBnestceBlibert?BcampagneBamisBtraitsBsphinxBrageBpoingBnesBmornesBdigneBdemeureB	frissonneBbecBvingtBsanglotsBfi?vreBfielBberceBbrancheBvillesBtaimeBsavantBnoiresBlodeurBdentBcuivreBcrainteBcharg?Baust?reBvisionsBpuissantBpropreBmontaitBglisseBfiersBerrantBdiraitBtristesBrongeBpr?treBpenserBmaisonsBl?gerBlenteBjeuBjambeB	invisibleBfa?onBd?toilesBdroitsBvouloirBsommeBrudeBp?leurBpatrieBmomentsBmielBfamilleBdoubleBdabelBcolombeB	blasph?meBvertigeBteintBmajest?Bvo?teBp?ch?sBplongeBl?geBloc?anBlaspectBfamilierBca?nBberceauBtableauB	spectacleBserontBr?verBondesB	mouvementBinqui?teBhaisBgriseBfollesBcrini?reBchim?reBb?niB	tourmenteB	splendideB
silencieuxBpenchantBfr?lesB?trangesBvouluBtravauxBsommetBrideauxBmauditsBloisB
lesp?ranceB
brouillardBanciensBaffreuseBviergesBti?deBspectresBjuilletBfen?treBdivinsBbonsBtenirBsueurBsubtilBl?preBlysBenferB
souffranceBquestceBnavezBl?t?BlonguesBchantentBallezBterreurBsableBros?eBrondeBrampeBpontBlacBdel?BchansonsBcerveauB?pouxB	vaguementBuniqueBrobesBpo?sieBpareilleB	papillonsBneufBmontsBmettreBcharmesBamersBverreBtant?tBsereineBjet?Bid?eBhorsBg?antBgaucheBfeuillesBd?sertsBcr?ationBcourageB	chevelureBbi?reBvertuBtournerBtireBs?teintBroulantBregretBpara?tBmangeBfangeBvaincuBsongerB	rossignolBravieBlouraganBgrandesB?critBprisonBperleBmuseB
limmensit?BjeuxBdrameBchiensBbordsBappasBseraiBpo?tesBnocturneBmauxBfusB
formidableBfid?leBdureBcouvertBaucunBonglesBmortelsBlentBlaitBcruelBcoulerBboueBtrouBsimpleBsauleBrythmeBm?talB	flambeauxBcruelleBveulentBsoinBlimmenseBgrandeurBgrainBc?teBconnaisBbatailleBvoleBsouvreBsoieBr?clameBjusquesBcr?neBchoeurB?clairsBtortureBrouxBrondBp?meBprennentBpassentBmarbresBlyreBlivideBlantiqueBensembleBcrieBcoucheBconfusBvibreBreinsBluireBlivresBdresseBdautreBclochesB?BtailleBpassageBlantreBbourreauBantreBvraisBsenfuitBreposeBlasseBlaidBfum?eBfr?resBfrissonsB	attendantB
tranquilleBsaintesBparesseBlaissentBjeterBhumideBesclaveBcerveauxB?aBvenezBtr?veBsourdsBr?pondBquellesBprierBnommeBloupBinfernalBgerbeBfouetBfataleBdhiverBbless?BvoeuxB
vieillardsBvictoireBriantBrempliBpieuxBmaladeBmaigresBlamenteBlameBlaisserBdamn?BcristalBcoupeBbont?BbanalBvoieBdessinB	conna?treBchaosBchagrinBaugusteBaimonsBvirgileBsillonB	regardaitBnoteBm?chantsBlennemiBjettentBd?monBdonnerBdemandeBcrimesBbonnesB?panouieBvivanteB	rayonnantBpaysageBmidiBf?tesBcaressesBavideBsillonsBsaisitBm?resBl?clairB	faisaientBd?b?neBconna?tBchocBsourceBrichesB	poussi?reBportantBnomsBixBvisionB	quimporteBplaieBperdusBoc?anBinfiniBfaustBennemiBcoursBamanteB?t?B?BvenaitBrimeBnordBmonterB
monstrueuxBchersBbrinBverteBs?veBremordBminuteBminuitBlamantBfilBderni?reB	contempleBcit?sBbatB
tristementBsuaireBquoiqueBgranitBfruitsBfolieBcriantBcertesBauroreB?preBromanB	regardantBprixBnasBfanfareBentendreBehBconsoleBcit?B	cauchemarBblondsBposesBplombBpassionsBlid?alBglaiveBeutBd?go?tBadorableB?paulesB?paisB	travailleBr?gleBracineBloreilleBlontBguideBgazeBfra?cheBfiniBd?baucheBdamn?sBcenBvainsBsortentB	singulierBperdreBmontentBgriffesBfoudreBcombatsBclaireBastuBsourisBsilsBpensersBmourantBmembresBhonteBflotteB	d?sormaisBdursB
dautrefoisBblessureBvigneBunsBsuppliceBsorci?reBsaisonBsagesseBprinceBpourraitBplumeBmontantBlautomneBhautesBfi?reBexquiseBchim?resBauraitBassiseBsenivreBsacr?BroseauxBrentreBquelsBinfinieBennuisBbizarreB?prisB?clairBsourcilBsermentsBrefletsBreculeBpiliersBparlaitB	effrayantBvolerBouvertsBnoeudsB
michelangeBjeanB
immortelleBfrissonBdisentBcausesBb?tonBvoulezBtrainBstupideBsatinB
pourritureBnoblesBm?lancoliqueBid?alBfurieuxBfrappeB
comprendreBchoisiBchastesBbrunBaim?BsoufflesBriresBridiculeBpench?BmuetteBmentBlumineuxBferaitBentierBvinsBtonnerreBtapisBsoifBserpentsBsecouantBpr?sBminceBmartyreB	lescalierBf?condeBfleuritBestilBchaleurBbaiseBao?tBveuveBsuBriaitBparcBnavaitBmuscBmati?reBmariBimageBfortuneBferaBdisantBdiraBchuteBtueBprom?neBplanteBpeintreBparfum?Bl?hautBflamboieBdiamantBc?t?sBcompliceBch?nesBb?nieBbuissonsBaimezB?parsB?jeB?cestB
voulezvousBtrouv?BsecBruisseauBregardezBqueueBnageBmaintBgrondeBglorieuxB	farouchesBennuiBc?lesteBtonsBsenvoleBsacr?sBreluitBmontagneBmarquisBloeuvreBlambeauxBfl?chesBfigureBfasseBdoucesBcourirBcourBcorbeauxBcalmesB?blouiB	voyageursBvoeuBviveBvaseBvampireBtempesB
splendeursBsourdeBmavezBloisirBjetantBf?eBcombatBch?resBblondesBbasseBardeurBp?leBp?seBproseBnulleBmarieBlaissantBironiqueBglaceB	d?trangesBdoreB
contemplerBbris?BtaireBs?v?reBsacr?eBrayonneBpav?BnombreuxBmarqueBifsBgaiBdoisBdiscoursBdehorsBcontourBchaudBcercleBcarreauxBcageBallumeBxB	vainementBsouvienstoiBrayonnementBpr?BpoignardBpl?treBpleinesBperdBjentendsBhistoireBgr?veBf?tesBemplitB
douloureuxBcomprisBchaudeB	cavaliersBb?tiseBbuBbrunsBaveugleBarceauxB?leBtra?nantBsurpriseBpliBmontrerBmensongeBlhumbleBflambeB
courtisaneBcontientBcinqBcaliceBazurBvoudraisBsupr?mesBsinistrementBseizeBsarr?teBperceBpeintureBnu?esBnayantBmestBlid?eBlautelBjalouseBgrimaceBb?antBboitBaspectBaim?eBvertesB	trompetteBtournantBsourcesBrareB	puissantsBpageBnuesBm?l?sBmangerBlampesBlaissaitBicibasBhymneBhibouBhaitBclocheBaur?oleBaimantBtrompeBtraceBtigreBsonnetBpr?f?reBplumesBplatonBmontraitBgr?leBgarderBfardBc?sarBchargeB
chantaientBcaveauBarriveBardenteBsaleBruinesBquavecBpoisonsBmortelBlorsBlardeurBfunesteBfuisBfausseB	dailleursBclousBchr?tienBbalanceB
?tincellesB
triomphantBselonBrapha?lBm?leBlobscurBjyBinconnuBherbesBhabiteBclocherBavezvousBattendBardentsB1842BvoulaitBvitrauxBsaphoBposerBpla?tBmauvaiseBlitsBhagardBfersBd?corBdiscretBceuxciBcadreB?blouissantBversentBtomb?BtalonsBsongesBsommeilsBrapideBp?resBpremiersBn?treBn?treBm?rBl?gliseB	impr?gn?sBf?roceBfinsB	fantasqueBeuBcacherBarri?reBaigleBvioletteBt?cheBsongezBpolisBplaintesBmarchentBlaubergeBjouerBglac?Bgard?BferaiBfauteBdherbeBcoussinsBcollineBchairsBb?tailB	bienaim?eBtroublerBsorsBpudeurBpoingsBpassaitBm?lerBmasquesBivresseBgonfl?Bd?menceBcourbeBcharogneBappelleB?troiteBv?tuBrevientBquauxBpleuraitBouverteBnouvelleBmerciBlhuileBlettreBlacsBlabriBjallaisBfurentBenvoieBd?cembreBdragonsBdivinesBdhorreurBdansaitBcueillirBcommenceBb?cheBv?tementBuniversBsaventBsagesBriveBpuniBportraitBplaneBphiltreBpassantsBnouveauxBmerveilleuxBlaiBjeusseBjambesBignoreB	d?sespoirBcoinsBchaudsBcentreBb?h?motBbrusquementBbarbeB?tiezBvictimeBvaincusBs?veilleBna?fBmauditeBlabeurBhautsBgonfl?sB
fr?missantB	d?licieuxBdevenirBdegr?sBdailesBc?lestesBconfus?mentBcaverneBbouquetsBbiseBagileBaccordsBvolantBvilsBvicesBtontBsonsB
semblaientBpoudreBpleurantBoctobreBn?BnoyerBmonumentBmanqueBivresB	hippolyteBgracieuxBfun?breBfuiteBfaisaisBdocteurBcharit?BxxBxvBv?cuBt?moinBtr?sorsBtordantB
taisezvousBsph?reBsoucieuxBsentiBsculpt?Bpr?cieuxBnoseBmonotoneBlinstantBlhumaineBjugeBgrillonBespoirsBd?voreBdonn?B	descendreB
compagnonsBbalconB?perduBxviiiB
vermeillesBtendentBsenteursBsecoueBsanglantBquitterBprogr?sBpailleBjusqueBillusionBgriffeBfragileBfisBcollinesBchasseBceciBboeufBangoisseBallaBxviBtoucherB	surveilleB	septembreBsavantsBr?pandBrideauBretireB	regardaisBp?ch?B	pardessusB	m?choiresBl?tangBl?clatB	lhistoireBlavenirBgosierBgaisBfauvesBd?chireBdormaitB
cr?pusculeBxivBtressesBterrasseBsabreBruineBrugitBrivagesBpr?teBpendBoublierBondeBnastuBmortesBlorientBhabitsBfumantBfinirBessorBd?licateBdonnaitB	couvercleBcherch?BailleursB?oBxixBsiedBseineBseffareBr?jouieBr?duitBresterBregretsBpoidsBplaireBoubli?BobscursBmoroseB	merveilleBmarcoBl?th?B
l?paisseurBjouantBhaillonsBfuiBfouetteBfineBeffroiBdevoirBcorB	caressantBbourbeuxBbiensB	attitudesBvueBtroubl?Btouch?BsonnetsBsanglotBplainesBpeintBpanBmoindreBmeursBmamelleBlouisBlaigleBironieBinf?mesBgouffresBga?mentB	cherchentB	blancheurB?tionsBxiiB
voluptueuxBvivesBvertusBsc?neBsallumeBsachantB	plusieursBpeuxBoubli?eBm?lentBmourezBlouangesBlhonneurBlangageBjusquauxBjuponsBingratBgranditBfrileuxBenfersBempireBdismoiB	cercueilsBcarnageBblocBblafardB
bataillonsBapp?titBantiquesB?choBvibrerBtorrentBtenantBsurprisBsublimesBsouffertBsauteBsagiteBquitt?BprodigueBpo?meBpluvieuxBm?laientB	mollementBmeilleurB	madeleineB	l?viathanBl?g?reBfr?mitBexquisBd?licesBdautantBcourtisanesBcoulentBchoirBcach?sB	brillantsBbattreBarmoireBappara?tBallaientBaigreBtremblerBtournentB
squelettesBsententBsavezBsaturneBr?gneBreditBpritBparlantBmont?BmoireBlierreBlecteurBlangoureusesBhaillonBflaconBerreBdurerBchevetBcellesBcath?dralesBcarcasseBbr?lantBbrutalBbrumesBbl?sBbleuesBapp?titsBaim?sBailleBagonieB?geBxviiBxiiiBverbeBtraitBtoucheBtombantBtimideB
semblablesBr?deBquaisBprenantBpouvionsBperruqueBparolesB	montagnesBlalc?veBjupeBjardinsBfureurBd?pitBdormantBcompteBcharBcass?BbeaucoupBaiglesB	tr?pass?eBthymB
tendrementBtamboursBtachesBsourcilsBsant?BsangloteBrevenirBna?treBmiracleBl?normeBlenfanceBlancienBjanvierB	flottantsBfileBestuBessaimsBerrentBdormentBdarbresBcrierBcourseBclart?sBcilsBagiteB?clatBvoistuBvip?resBvapeurBsiffleBsalesBpleurBnonchalanteBnimporteBm?neBmolleBmienBlignesB	lhumanit?BglaivesB	feuillageBdedansBcr?peBcort?geBciergeBch?tifBchaudesBchatsBchapelleBcach?BborneBaubeBartBvatenB	tendresseBsuffitBstupidesBsallongeBsaisonsBrumeurB	profondesBpipeBpeintsBpapillonBnefBl?treBhumainsBgazB
flamboyantBferveurB	escaliersBenchant?Bd?lireBcadavresB	bourreauxBvolont?BviventBstatuesBserreB	sendormirB	sacrificeBr?verieBregarderBprobl?meBparfaitB
magnifiqueBl?galBluisantsBlhymneBivrogneBheureuseBgrinceBgrecBfi?vreuxBfiltreBfiBdeversBcroupeBcha?nesBbougeB	agonisantB?tonn?BxiBvenusBtordBsortantBsavaisBsacBr?v?leBrougesBpuisseBpleurentBpeindreBmorosesBmitB	meilleursBlempireBhonn?teBfleurirBerrerB	emportantB
effroyableBd?chirerBdesseinsBdeffroiBdansantBdangeBcourantBcaillouxBaviezB?coliersBviolonsB	vaisseauxBtinteBtenaitBtenBs?crouleBsurgirB
singuliersBrocheBrocB	pouvaientBpouss?BpointeBpeigneB	palpitantBmollesBmerlesBmarinsBlustresBleffroiBlatinsBjoyauxBenchanteresseBd?lugeBd?dainBcypr?sBcorbeauBcimesBchocsB	caressentBcachotBavareB?chapp?eB?treBtirerBspleenBsentierBr?veursBr?pondsBroyalB	proph?tesBphoeb?BpeutonB	passaientBpalmiersBmuetsBmarchantBluisentBinconnueB
implacableBgr?vesBfoinBentenditBd?liceBcoupableB	couchantsBbl?BbecsBa?euxB	?chevel?sBtr?vesBtroubl?sBtoiletteBtenteBsujetsBspiraleBsouviensBsoul?veBsongeurBsablesBrimesBquaB
porcelaineBoublieB	nouvellesB	notredameBmontrantBmarchaitBluxeBhouleBf?tBfroidesB	dentellesBdaimerBbais?BurneBtombentBspiralesBsoupireBsouliersBr?alit?BrudesBrest?B
resplenditBpartentBorageuxBoasisBm?tauxBm?diteB	muraillesBl?choBlopiumBjaillirBhorizonsBhagardsBgloiresBgenreBfauteuilBfaudraBentendsBd?licatBdistraitBdifformeBdicibasBdiaphaneBdalb?treBcro?treBclimatsBchoisieBberc?B?troitBt?chantBtra?nentBtombesB
solennelleB
rossignolsBrendaitBquerelleB	promenantBprieBportBpompeBpleutBpattesBnainsBm?tsBmillionsBlourdesBlorageBloisirsBlavoirBjaillitBhonneurBguetteB	guenillesBferaijeBd?partBdemiBcourrouxBcoucherBch?neB
charmillesBbouquetBbergerBbandeBanimauxBamieBamasBvigueurBtransparentsBtiresBsoyonsBsoucisBsondeBsingeBportailBployerBpenseurB	paresseuxBpalmesBpaisibleBorgueilleuxBobjetsBmartyrBjetBfauneB	faubourgsBdrapsBdoiseauxBdhierBderniersBdanserBcyth?reBcourentBcoquetteBclaironsBcascadesBbriserBbl?mesBbagneBaiB?coutezBvivaitBvictimesB
vainqueursBtrouventB	terrestreBsinonBsauraBsablierB	ruisselleBravitBquestBquapr?sBpossibleBphrasesBparlaBm?reBmeublesBl?tudeBl?nigmeB
instrumentBimpureB	hasardeuxBgrassesBgar?onBfl?trieBenivreB	encensoirBdupeBdrapBcrinsBcharbonBbuissonBbrebisBambitionB?ph?m?reBtombaitBstrophesB	sanglantsBrepentirBouvraitBnudit?BmoucheBmorceauBmadoneBl?cumeBlugubreBlessaimB	indolentsBhurlantBhiBglaciersBfrissonnantBfaisonsBernestBentrerB	entendstuBembaum?eBd?sol?Bd?meBdenferB	dangereuxBcourb?Bcl?menceBchariotBbonheursBblocsB	blessuresBantresBalc?veB	?tincelleB?ternelsB?gliseB?galB?clatantB?etB?elleB	v?ritableBvoulonsBvip?reBvifsBvictorBtuerBterresB	taciturneBsonoresBsigneB	savezvousBroulentBromainsBpuisquilBporterBplanchesBplaintifBouvrierBoseraBordreBn?tantB
mouvementsB
ma?tressesBlivresseB	laraign?eBlanguesBlalb?treBimpurBhugoBgestesBfinitBfautilB
entraillesB	d?sastresBd?mesBdixBdenfantsBcouteauBbusteBboutsB	bataillesBallantBaboisB?normesBxxiiiB	vagabondeBsongeantBr?seauxBrouesBrochersBretrouveBrestonsBplafondsBpavaneBparfum?sBnoeudB
margueriteB	lointainsBlaur?oleBlatmosph?reBjuifBgroupeBfl?teBfid?lesBfa?teBd?funtBdrapeauBdodeursBcruelsBceluil?BceintureB	balan?antB
accompagneB?toil?eBtremblaientB
transportsBsotsBsi?clesBrouetBrivi?reBp?tureBpromisBpotBmiasmesBmesseBmasBlinspirationBlieuesBjourn?eB	illumin?sBgrilleBgazonsBfautesBdonnentBdiciBdallerBcurieuxBclimatBchauvesourisBchauveB
charmantesBbraveB	bouteilleBbattantB	ang?liqueBaimentB?l?mentsBtourbillonsB
tourbillonBtorchesBsouriantBsereinsBremueBrefluxB	reconna?tBrancoeurBram?neBquontBquenfinBpinsB	paupi?resBpassezBparentsBorgieBnaurasBmyst?rieuseB	mystiquesBmentonBlieB
lexistenceBinnocentBinfinisB	inconnuesBimmensesBgr?BgibetBfrancsBfol?treBfassentBfadeBdiversesBdhommeB	cherchaitBcaveauxBboiteuxBbleu?treBbelBbattuB
amoureusesB?treintB?closeBxxiBv?treBvermeilsBtrou?sB
tremblantsBs?pulcreBsymboleBsouvrentBrespirerBreprocheBpesantBpenduleBpaysagesBoreillesBm?rsBm?mesBmoiti?BloueBlotusBlimageBlentsB	langueursBlamanteBlactionB	ineffableBinclin?BfermentBfaucheBextasesBdeuilsBdestin?eBc?lineBcroyaisB	compagnonBchapeauBcertainBbrouillardsB
blasph?mesB	anciennesBaigresB?claireBvraieBvitreBtr?nesBtransiBtourterellesBtisonBsoufflerBsoir?esBsinc?reBsalonBr?veilB
reviennentBp?trirB	puissanteBpr?tBproposBpeuplesBpensifsBmouchoirBmiseB
miraculeuxBmendiantBmaimerBmachineB	lharmonieB
langoureuxBlangoureuseB	laboureurBjointBhalliersBg?antsBfuyezBfr?missementBflatteBdauroreBdateBdahliaBc?l?bresB
cath?draleBbr?lanteBbougesBbancsBarmureB?moiBvallonBvaineBvagabondBtroncBtorcheBsoldatsBsoinsBr?vaitBreluireBpatteBnaijeBnagentBmoribondBminutesBmeurtriB	laissemoiBjoueurBgueuleBfrivoleB	fra?cheurBfestinBestelleB
entrouvertBendormieBellem?meBeffetBdevineB
despotiqueBconseilsBcomprendB	chandelleBchaiseBbrisesBbourseB	all?gorieB
aimonsnousBaccoud?B?clateBvoyonsBvolentBvolaitBvienstuBvaletB	tourmentsBterrainBsolennelB
silhouetteBr?elBruchesBp?cheBpr?terB	premi?resBpourpreBplainsBpauvret?BpanierBnu?eBmourutBleverBleffetBlauriersB	largementBlanceBimagesBhabitBgrotteB	fraternelBessaimBeffar?eBdoubliBdoiventBdivresseBdictameBdiamantsBdestinsBcroisantBcessezBcervelleBcerteBblafardeBbaigneB	angoissesBabsentsBst?rileBsonnantBr?seauBrentrerBrasBramuresBp?lem?leBpouvaitB	portraitsBpieuseBpierrotBpatoisBmeureBma?tresB	marchandsB	mandolineBmajestueuseBlaiss?BhuitBhancheBgueuxBgrasseBfr?mirBfroisseBfran?aisBflotterBflaireBfixeBetreBembaum?BdaisBdBcourtsBcompagneB
chevaliersB	chaumi?reBcentsBbienheureuxBanimalB	admirableBab?mesBvoyaisBvendangeBtravers?BteinteBs?reBs?lanceBsoupleB	sabattentBr?veuseBroulerBrauqueBquelquesunsBpoussantBpassaBn?taientBne?tBnestuBna?fsBmang?B
lointainesBlionsBhumblesBhanchesBflamboyantsBfixementBfestinsBdennuiBdencensBdemanderBcouventB	connaistuBcolonneBcherchezBchasserBcelluleBcasseBcantiqueBbrideBbassinBbarreBbarqueBbainBaffol?Bador?BaccordB
?ternellesB?taitB?jaiB?aBvainesBt?tuB	terriblesBterniBtenaceBs?tonneB
souveraineBrideBp?n?treBp?liBpr?pareBpresseBplong?BpendentBpavillonBpardonsBoseBmadrigalBlutteBlibertinB	lassitudeBlargentBlarcheB
laissezmoiBlaideurBlagneauBjaisBgrainsBfatalit?B	fantaisieBd?couterBdorgueilB	descendezBcr?susBclaironBbr?lerBbilletsBbacchusBautelsBaiseBaiguilleB?ternit?B?teintBz?nithB	v?tementsB
volontiersBvivezBtendantBtendBtadoreBsuiviB	solitudesBsir?neBsimplesBseraisBsendortB	sculpteurBsallieBsaisirBrochesBrevivreBrabelaisB
profondeurB	portiquesBpointesBpapierBobliqueBobjetBna?tBmurailleBmarinB	laventureBjignoreBjalousesBiraB
int?rieureBhumainesBhiversBgr?lesBgothiqueBglissantBfran?oisBfl?trisBfiletB
feuillagesBfatigueBfabuleuxBendormiBemporteB	d?sesp?reBd?bileBdrap?Bdor?sBdentreBculteBcadenceBb?anteBbuvantBboudoirBatroceBanBajouteBaimantsBaigu?B1840B?vousBwagonBvilesBvautoursB	triomphalBtapageBtaiBs?chappeBsyllabesBstyleBsph?resBsouffrezBservirBsecsBsabotsBrubensBreviensBrecevoirBrangsBrangBpuanteurBprofilB	pi?tinantBpartiBparlentBparentBmeuteB
l?ternelleB
longuementBlimpideBlestBinutileBing?nuBhumili?BhauteursBhabitantBfoulesBferventsBfen?tresBfaubourgBfallaitBesp?ceBd?meBdurantB	difformesBcouronn?BclochersB	ch?rubinsBch?vreBb?nitBbr?l?BboucherB	?clatanteB?chineB?baucheBxxivBvoletsBversemoiBversantBvastuBs?taitB	surann?esBsouffrirBsem?BsecondeBsapaiseBsachetB	retrouverBrespireBremplisBravisB	quiconqueBpourraB
n?cessaireBnuqueBnourriBmyrrheBmuguetBmouchesB	monumentsBmontonsBl?therB	loreillerBlilasBlarcBlaiseB	innocenceBheurtantBhanteBfra?chesBfoul?BfondreBexil?sB	esp?ranceBd?robeBdistuBdhoraceBdespritBcuireBcr?eBcommunB
clandestinBchouetteBcampBcalvaireBbutteB?t?sB?tendBxxiiBtrameBso?lB	soulevantBsenvolerBr?citBrumeursBrubisBroyaleBrondsBreprendBray?BqueuxBpouvaisB	phosphoreBpassesBoreillerBnerfsB	moustacheBmorsuresBmagiquesB	lendemainBignobleBhouxBfloconsBfiguresBexcitantBdressantBdrapeauxBdesseinB
descendentBdastresBdamesBdaffreuxBcyniqueBcro?tB
connaissezBchoixBchineBchauffeBchariotsB	bouvreuilBbless?sBbizarresBartistesBaimaitBaimablesBvoyaientBveinesBtriomphanteBtoutpuissantBstyxBsouriaitBsoupeBsond?BseraientBsc?neBsalu?Bre?oitBreprisBreposerBremonterBpousserBpolaireBpleuvoirB	plaintifsBpi?geBperdueBperdantBobstin?BnourrirBnentendBmillionBmarsB
marchaientBmaintsBmaimeBluxuresBlhorlogeBlaffreuxB	j?coutaisBinsecteBinou?B	innocentsBinformeBh?teBhiverBhirondellesBgardienB	fouillantBfentesB	extatiqueBentendezvousB	enseignesBdressentBdisaientB
dhorriblesBdevoirsBdanteBdaileB
conscienceBconcertBciseauB
chr?tienneBcha?neBchatteBcachantBb?atrixBbraiseBbarreauxB	attendaitB	?touffantBveillentBvautourBvant?BtripleBtriompheBth?r?seBtambourBtaill?sBsouffletB	soufflentBsouBseulesBservileB	sentimentBsaulesBsasseoirBsabbatB	r?verb?reBr?citsB
r?chaufferB	r?chauffeBretourneBrendsBredoubleB	rayonnaitBprodigeBpouvonsnousBpourvuBpeuplerBper?antBpendaitB	parlaientBparisienBouraganB	ornementsBorateursBmo?seBmouranteBmeurtrisBmassifBlutteursBluisaitBlinsecteBlinconnuBlexilBlatinBlangleBj?suschristBjureBjavaB
insensibleBindiff?renceB	guerriersBgrosseB	girouetteBgaieBfluxB	d?sh?rit?Bd?placeBdextaseBdevezB
desp?ranceBdelphineB	cuirassesBcueilleBcr?nesBcriezB	couronnesBcorsetBconsolerBconseilBcol?resBcibleBcarminBcaravaneB	boucliersBbergersBautelB	approcherB
appesantisBactionB?l?veB?galeB?couteB?closBvoulutBvoltigeB
versaillesBtournoieBsuiteBsouvrirBsotBsoiBsentiersBsentaisBsecr?tesBsecoursB	scintilleBrestaB	renouveauBrailleurBquautrefoisBpostureBportaitBplatBornementBnuanceBmorduBmoqueBmarchaisBlencensBjusticeBherbeB
harmonieuxBhainesBgrottesBexcept?BenseveliB	embrasserBd?tincellesBdocileB	distraiteBcompletBcommisBcombl?BclosBcloisonB	classiqueBch?teauB
christopheB	chevalierBcasqueBbr?lantsBbrunesBbrod?BbouleB
avalanchesBall?Bail?Bador?eB	?clatantsB?leB?ilBvibranteBvers?BvapeursBunieB	tr?pass?sBtomb?eBtempeBsubitB	somptueuxB	sinistresBsalleBsaigneBsacheBr?v?laBromanceBreversBrenduB	rencontreBrefl?teBp?leursBprudenceBprologueB
prodigieuxB	princesseBplaintBpartsBpagesBodieuxBnavaisBmoinesB	meilleureBl?coleBl?chelleBl?chesBligneBlhirondelleBlazareBjugerBjouitBhurlerBhardieBgrabatBglobesBglac?eBgagnerBforc?BfleuvesBfibreBfendB	faiblesseBenviantBeldoradoBduresBdor?eBdoeilBdianeBdhommesBdenfantBdansentBc?tesBcuisantBcriaBcrainsB	coulaientB	corneilleB	ch?timentBchanteurBchantesBchagrinsBcasquesB	carrefourB	campagnesBbruteBbornesBbaumesBballesBauraBaucuneBapr?sBamoureusementBallureB
absolumentB1831B?teinteB	?pouvant?BvoulantBtransparentBtorsBtoim?meBth?ba?deBs?tantB
s?pulcraleBsisinaBserezBsaitonBrepasBreligionB	prendraitBpoussentB
pourraientBpoumonsBpensiveBpenchentBpeign?sBpatienceBpartirBparfaiteB
palpitantsBoubliBnagu?reBnageantBm?l?eBm?decinBm?chanteBmirantBmineBl?pouxBlodeBleffortBlartisteBlarmureB
laiss?rentBjauraisBironiquementB	inf?condeB	illustresBillustreB	illuminerBidolesBheurtentB	heureusesBgerbesBgangeBfracasB
fantasquesBenfantinBd?ploreBd?corsBdyeuxBdessusBdagateBcorolleB	corbeilleBcharmerBcachentB	bourgeoisBbesogneBaugmenteBattirailBancienneB?gauxBtenterBsottiseBsignesBsavanteBsaintet?B	ressembleBrel?veBramierBrailleB	quimport?B
quavezvousBp?litBp?lirB	puissanceB	prodiguesBprocheBpoliBpalmeBpactoleBn?sB	n?nupharsBnourritBnavonsBmorbideBmoissonsBmoeursBmesureBlorgueBlogeBlinBlanguissammentBjugementBirrit?BintimeBinsulteBinsolentBinquietBgrecquesBgantsBfrangesBfouill?BfouillisBflottentBfilerBfatigu?B	entendaitBeffar?B	d?risoireBdisparuBdispara?treBdamneesBdamerBcruaut?Bcrois?sBcrayonsBcoupoleBcoudesBcomprennentBcollierBclo?treBchenusBchefBchaleursBcellel?BcapriceBbrillaitBbabelBaveuxB?talantB	?lastiqueB	virginit?BvermineB	vendangesBvaporeuxB
transformeB	s?panouirB	statuaireBspirituelleBsifflaitBsaluerBromancesBrefl?t?BraconterB
quenouilleBporchesBpindeBpeupl?BperdonsBpendantsBpench?eBnotesBnarineBmirageBmignonBmeurentB	mensongesBmaladieBmainteBl?pongeBlifBlettresBlagonieBjoiesBjaper?usBironsBimmobileBhu?esBheureusementBhardisBgu?peBgalantsBgalantBgBf?mesBfraiseBfl?triBfauvetteB	explosionBesclavesBeffac?Bd?serteBd?esseBd?cro?tB	dormaientBdonnezB	distraitsBdeumBcouvreBcontiensBcontentsBclefBclairesB	cherchonsB	chancelleBboileauBbl?meBbeauteBauraientBattendriBardentBadoreB	accroupisB?toufferB	?pouvanteB?pouseBvusB	voyezvousBviceBverd?treBtravaillentBtaitBs?taleB	sortaientBsexhaleBsauvagesB	sauraientBsaignantBruesBroutesB	repentirsB
rafra?chirBp?dantsBpoilsBperversBpenduB	parfum?esBpardonneBpardel?BnovicesBm?taitBmyst?resBmouvantsBmiennesBmarteauxBl?cartBluniqueB	luisaientBlouvrierBlivrogneBlingeBlimaginationBlass?B
larabesqueBlaimeBlabb?Bj?couteBitalienBinsens?BhurlentBhumanit?Bhom?reBgrenierBf?condesBfortesB
floraisonsBentrouvrantBd?voquerBd?tourBd?funtsBdress?eBdrapeBdouteuxBdentelleBdardB	courtisanBcouplesBcontemplentBcontemplationB	cl?op?treBchassantBcaptifBbondsBblanchitBaubergeBarriverBappelBanneauxBadieuxB?crinsB?critB?lesBvoletBvileBverraBvatilBvallonsBtimbreBtellesBtaudisBs?r?nit?BsuspendBsoufreBsorstuB
simplicit?BsentisBsentantBsentaitBsenfonceBsapinB
rhythmiqueBretentitBretentissantBrequinBrefrainBrapidesB	p?n?trantBpublicBplaisBpiteuxB
pervenchesBpass?sBouvrentBoutreBorn?BorageBnappeBm?lancoliquesBmourantsBmienneBmettentBmalheursB	l?chafaudBlouestBliti?reBlimmobilit?BlenversBlavaitBinond?Bhant?BgrecsBgravesBgigantesqueBgenouBfurtiveBfumeuxBfrotteBfor?atBforgeBferm?sBferm?BfermaiBenvieuxBennuy?Benferm?BenchantementsBd?mesB
d?vouementBd?tresseBdordreBdopaleBdoB	dhumanit?BcirqueBchant?BcarreauBbuffleBbouesBblondBbalanc?sBattitudeB
atmosph?reBarcheB
andromaqueBall?eBacteB1872B?clair?B?aBxxvBvoltaireBverrezBus?B	universelB
t?n?breuseBtenezBtaureauB	s?lan?aitBsouviennentBsoumisBsongeaisBsoifsBsacreBr?sign?eBr?ponditBrougeurBrosiersB
retournantB
puissantesBproscritBpo?tiqueBpoudreuxBpointsBpocheB	plongeantBplaignezBpench?sBoffrirBnicheBnestilB	m?nagerieB
mis?rablesBmeulesBmenivreBmelancholiaBmaturit?BmariageBmanchesBmacabreBlouangeBlordreBliquideBlimonBlhippopotameBlentourBlaverBlaurierBlanterneBlancesB	jongleursBjiraiBhibouxBheurteBherculesBgrandirBfoetusBflaconsBfianc?eBferaientB	expiationBensanglanteBeffar?sB
d?chiffrerB	d?carlateBduvetBdivaBdeschyleB	daust?resBdaujourdhuiBdalarmesBc?sarsB	contempl?B	colombineBcolombesBcendresBcavalierBcadresBb?leBbrumeuxBbijouBber?antBbaignaitBann?esBalbertBafriqueBaffreusementB?clotB
v?n?rablesBvoyagerBveuvesBverserBverdureBveninBtournaiB	tonnerresBtoisonBtermeBs?meBs?cheB	suspendreBso?leB
souterrainB
simplementBserr?BsappelleBsachezBr?p?teBr?glesBrocsBrobusteBrevolerB	retraitesB	publiquesB	pr?tresseB
prostitu?eBpri?BportiqueBpoilBphilippeBpendreBpencherBparlezBparalytiqueBnoyaitBneufsBnaseauxBm?lantBm?nageBm?g?reBmorsureBmontezBmonstrueuseBmoineauBmenteurBmagiqueBluisantBlirr?sistibleB	libertineBlerreurBlenvieBlap?treBjouaitBjet?sBjaunesBhonteuxBg?mirBgu?reBgonflantBgardentBfrimasBfarcesBentour?B	ensorcel?BencensBd?votionB
d?sesp?r?sBdompt?BdisleBdiraiBdessinsBdartBcrucifixBcroyezBcroiraitBcouvertsBcontenirB	conserverBcolliersB	cocotiersBcirculeBch?vresBchaufferB	charmilleBcerfBb?cherBbrutaleBbrillantBborgiaBaussit?tB1835B	?ph?m?resBvivantesBvendreB	vagissantBtuezBtremp?eBtorseBtombaBs?raphinBsuivieBsuaveB	stup?faitBstropheBsculpt?sBscrupuleBsabreuveBretiensBrestaiBrentraiB	renonculeBregard?BramiersB	poursuiviBpleureurBplansBperfideBpenseursBpav?sBparfumeBparasiteBpalpiterBnaturelBmorbidesBmodesteBmignardBmettraitBmarteauBmamellesB	l?treinteBl?pauleB	lointaineBliseB	lhorribleBlevantBlassautBlambeauBjalouserB	irr?solusBidoleBhabit?Bg?tBgrimpeBgoutBgourmandBgermeBgemmiBf?rocit?BfurieuseBfourmillantBfor?atsBfix?sBexpireBentra?neBensanglant?B
d?pouvanteBd?couvreBdisaisBdirastuBcroyaitBcourteBcoquetsBconvieBcontenteBcl?menteBcireBchass?sB
catholiqueBcahuteBbibleBberceauxBbalayantBbaisaitBaurontBartistementBaijeBabeilleB?taleB?mailB?criteB?batsBv?tresBv?mesBvoudraitBvivonsBvigilantB
vieillesseBvalentBtuniqueBtuesBtreizeBtreillisBtra?nerBtiensB	tellementBs?talaitBsuerB	sorci?resBsembl?Br?pondreB
ruisselantBroulaitBrougisBroucoulementBrockBrevenantB	rendaientBraconteB	pr?cipiceBpointuesBplant?BplacideBplacesBpenteB	pareillesBouvriersBnuancesBnatalBnaimeBmoussesBmorsBmontrentBmettezB	malicieuxBmaillesB
madr?poresBl?l?mentBlumi?resBlivraitBlaveBlappelB
laissaientBirrit?eBinterditBintelligentBinsens?eBinondaitBimaginezBignor?BgronderBgoujatsBf?esBfuturesBfranchirBfontaineBfl?cheBfavoriBesp?rerBengageBd?voileBdyBdavanceBdapr?sBdansesBconvientBconnueBcolosseBchassentB	chanteursB	captivantBb?antsBb?illeBbrigandBbaguesBarcBapprocheB?peronsB?quelleB?queB?monBvo?t?B	violettesBvillageBverduresBvendrediBvall?eBtriomphantsB
transport?BtitienBtisaneBterreursBtendresBtelsBs?paississaitB
symboliqueBsuisjeBsuccombeBsauveB	saignantsBr?pondisBromainBrevitBrenifleBreleverBreculerBpr?tresBpromenerBpoursuitBphareBpasteurBparutBparsB	panth?resBodeursBobstacleBnoy?BnoieBnageurB	musculeuxB	murmurantBmichelBmiBmarineBmaigreurBl?piBl?goutBl?gionsBlaccentB
jouissanceB	jentendisBimmortalit?Bid?esBhumaitBgueulesBglac?sB	gazouilleBgardezBfroncerBfresquesB	familiersBexpr?sBerrantsB	engloutirBenfanceB
effrayantsBd?filentBdonsBdiverseB	distingueBdiscr?teB
diff?renceB	dentendreBdadmirerBc?taientBcroyantBcourezBcourbentBcorsBcordeBcorauxBconduitB	condamn?sB
comprenantBcolsB
colonnadesBcigu?B	chlorosesBbronzeBboutonsBbaignentBattilaBarm?sBapais?sBalphabetBallemandB1836B?thersBvolcanBvisqueuxBvengeurBvalseBtr?flesBtraverseB
travaillerBtragiqueBtortsBtorduBtablesBs?parentBsoulierB
singuli?reB
shakspeareBsemainesB	sattirentBr?pandsBroyaumeBretentirBrena?tB	religieuxBredresseBrampantB	p?n?trentB
pr?cieusesBpress?Bplomb?B
pleuraientBpireBphryn?sBpeintresBpatientBpanacheBouvriraBoubliantBobscur?mentB	obscurit?BneuveBnaurezBm?priseBmurmuresBmezzetinBmar?eBmartyrsBluttesBluttantBlucr?ceBlorangerBlivrantBlinjusteB	lincendieB
lignoranceBlidoleBlessorBla?euleBlaveugleBlamesB	justementBjiraisBjaunieBinfid?leBhy?nesBhauteurB	granvilleBgeindreBgar?onsBfardeauBexilBessenceBeschyleBenvol?BenseigneBencombreBemport?Bd?fautBdominoB
domestiqueB
dobscurit?BditelleBdiscretsBdhom?reBdemandezB
daffreusesBcygnesBcuisineBcriminelBcouvraitBcouronsBcoudeBcouch?BcostumesBcontentBcom?dieBcolonnettesBclandestinsBchiffreB
cheveluresBchapeauxB	cervellesB
camposantoBbuvaisBboucBboiventBbizarrementBbeffroiBartisteBarcsBanglaisBalouetteBallumerBadamB
abandonn?sB1834B?ternellementB?tenduB?parseB	?galementB?colierB?chafaudB?ahBxxviBwatteauBvoil?sBvisiblesBvioletBvienneBvidesBvengerB
tr?buchantBtordreBtoilesB
tentationsBtemplesB	s?vanouitBs?railBsylphideBsultaneB	suborneurBsortitBsonnerBsongeaitBsaurontBsaisjeBr?v?Br?sonneB	r?pandaitBrustresBrouilleBrondesBrevuB
rendezvousBremplaceBreliquesBrapi?reBquainsiB	p?moisonsB	prosp?resBprisonsBpourrisBpourraisBpiteusementBpeinteBouvertesB
nonchaloirB	multitudeBmoyenBmarquiseBmadonesBlinjureB
lapparenceB
lamentableBlaissonsB
jattendaisBjasminsBinstrumentsB	infernaleB	indolenteB	incendiesB	ignoranteBhydreBhormisB	histoiresBharangueB	g?missantBgroupesBgisentBf?eriqueBf?condBfumierBfuiraBfr?miBferaisBfa?onn?sBerraitBengourdiB
emplissantBeffetsBd?trangeB	d?chirantBduelBdambreBcouvaitB
coquillageB	contenterBcong?BcomtesseBcognentBchemin?eB
centenaireBcausentBcalicesBbBalentourBaimestuB?lesB?tesvousB?tuBvoraceBvomissementBvivaceB	vert?bresBvertigineuxB	vermeilleBvaincueBt?tBtettentB	symphonieBsouriezB	solennelsBsifflerBsermentBsavaneB	r?pondaitBrondeurBrompreB
reviendrezBrepairesBregardaiBrefrainsBreconna?treBreconnaissezBrajeunitB	rabougrisBp?lisB	puisquilsBpsaumeBproph?teBprofond?mentBpos?BportezB
pi?destauxBpayerBpavoisBpartonsBpartezBpanseBoutilBogivesBobstin?mentBn?gresseBnoueuxBnocesBm?resBmouronsBmouilleBmoli?reBmatelasBmateBlirr?parableB	libertinsBlessenceBlentesB
lencensoirBlamentablesB
lambroisieBjumeauxBjuch?BjetsBitalieBing?nueBh?teBhallierBgondsBglisserBgigueBfrivolesBfrissonnantsBfrancheBfouillerBfleurieBflatt?B	flagell?sBfiolesB	fatidiqueB	espagnoleBentr?eBd?Bd?roulerBd?rouleBd?fiantBd?cr?pitBdiseB	dimanchesBdarbreBdangesBdaimBc?linB	curiosit?BcruellementBcroquisBcrisp?BcraqueBcrapaudB
confesseurBcognantBcloseBclameurBchim?riquesBchantezBchangerBcatulleBcarr?sBcaress?BcanonBbuveurBbrusqueBbrasierBbonsoirBbi?resBbienaim?BbanniBautomneBaspectsBap?tresBantoineBancienBanath?meBallum?BagitantBabandonB?tudesB?trangerB?talerB?mueB?muBvo?tesBvoilaBtressaillirBtrahitBtrahisonBtentureB	tarasquesBtapieBsurgitBsuantBsordideBsixtineBsiesteBsecr?teBsatansBsabotBr?veriesBrubansBroul?sBrong?B
romantiqueB	ridiculesB	reprochesBrel?cheB	rappelantBquamourBpuis?eBpuissesBpr?f?r?B	promenadeB	profundisBpouvezBpoussaitB	portevoixBportentBpomp?BpointuB	pilastresBpignonsB
philosopheBpeserBpersonnagesBpensivesBpendusBparesseusesBpardonn?BpalmyreBordureBneutBna?vet?BnacreBm?tierBm?tamorphoseBmyrtesBmouvrentBmorceauxBmoineBmen?BmaximeBmartyresBmancheBmaliceBmaimesBl?pouseBlovaleBlorsquilBlorangeBlibresBlev?sBlarcadeBlan?antBjoncsBjointesBjeusBinvolontaireB	hurlementB	hautainesBgu?ritBguivresBgorgesBfiert?B
fid?lementBfermezBenterreBd?videBd?truisaientB
d?sespoirsB	d?chirentB	doutremerBdouceursBdansonsBdann?esBdaiseBcruBcontinueBconstellationsBconcertsBciboireBch?tierBcabreBb?chersBbruyantsBbrockenBbrillentB	boutiquesBbourdonBbonjourBbavardeBazursBauriezB	attentifsB	ardemmentBarchangeB
appareilleBapparaissentBam?riqueBaboieB
?tincelantB?tangsB
?pouvant?eB?lansB?restonsB?laBv?tusBvitreauxBvenduBvaletsBunisBtourmentBtordaitBtimidesBtilleulsB	s?pulcralBs?l?veBsvelteBsuieBsouvientBsouffrancesBsocsB	serviteurB	seigneursBseffaceBsauv?BsalutsBr?ventBroyauxBroulisBrougeursBrev?tBrevoitBretraitBretoursBrepousseBremu?sBremplissentBremplirBregorgeBquitteB
pourraisjeBplatanesBplan?teB	pi?destalBpercheBpelouseBpavotB	palpitaitBoubli?sBopalesBoffreBobscuresB	nocturnesB	nevermoreBnectarBnauraitBnaitBnacr?sBmusclesB
multiformeBmomiesBmimporteBmidasBmeutBmasuresB
majestueuxBl?tableBlormeBlhaleineBlatineBlann?eB	laiguilleBjupiterBjattendsBinsult?BinsuBindigneBindigentBincoloreB
incertainsB	impatientBh?treBgrouilleBgrossiBgo?terBgo?teBgermerBgagneBfourrureBfixesBfinieBfeutreBfanfaresBenvol?eBennuy?sB
emplissentBempereurBd?robentBdorureB	distraireB	diaphanesBdhumainBdanseursBdacierBc?deBcriardeBcoutumeBconfiantBcommuneBcolibrisBcohueBcoffretBcl?mentBcl?B	chuchoterBchauvesB
chancelantBcass?sBbranleB	branchageBboudeurB	blasph?m?BbarbareBaust?resB	audacieuxBaspicBarr?t?BarracheB	aiguillesBagaceB?troitsB?tatB?coutaitB?clair?sBvisagesBvirginalButileBtr?pasB	tremblentB
traversantBtoupieBtonnezB	tomb?rentBtienneBs?jourBs?vesB	souverainBsosieBsoim?meB	sillumineB
sifflaientBseronsBsennuieB	sc?l?rateB
sattendritBsaphirBr?pandreBrevoirB	reprendreBrebordBravinBranciBram?eBrampantsBraisonsBquittentBquattendstuB
purgatoireBprofaneBpourrasBplantantBplaindreBpinceauB
peignaientBpariasBoursBormesBoffrentBoc?ansBnonchalammentBniob?BnaurontBnaturellementBnaquitBnaisseBnainBmineurBmincesBmenl?veBmenerBmappelleBmaitreBmacbethBl?corceBl?cheBlycorisBluthBlurneBlugubresBlorsquenBliqueurBlaugusteBlaquelleBlangesBlaisserallerBjouetBjauraiB	int?rieurBing?nusB
inexorableBinassouviesB
impuissantBimmondesBhuiti?meBharpeBgrossirB	gravementBgratteBgouttesBge?leBgambadeBgaleuxBfuyantBfureursBfourmili?reBforfaitsB	follementBfix?B	faisceauxBfablesB	existenceBexhalentBendortB	endormeurBemprisonnantB
embrassantBembelliBeffluvesBd?ployerB	d?go?tantBd?couleBdormezBdistinctementB
disaitelleBdiraisBdevintB
descendaitBdenbasBdaussiBdardaitBdaignaitBcuivresBcuisseBcr?pusculesBcriaitB	coursiersB	courageuxBcoupolesBcouperBcoteauBconteBconfidencesBcomposeBcomplotBch?rieBchoisirBchang?BceluiciBcandideBcabaretsBb?nirB
b?illementB
brillaientBbouclesBauraisjeBappareilBanalyseBaminteB	ambitieuxBalt?r?BaillesBabriB?p?eB?cresB?estceBvisiteBvasesBt?tusBtypesBtulipeBtrouvaBtricheB
tressailleBtravaillantBtranquillit?BtoutepuissanceBtigresBtaciteB	surhumainBsujetB	souvienneBsournoisBsonnaitBserasB	sentaientB	seffacentBsavanceB
saturniensBsaccag?Br?gnerBrivageB
rh?toriqueBrena?treBqueuesB	questionsBp?dantBpuisjeBpr?tsB	primitifsBprairiesBpiqueBph?nixBphraseB	peureusesB
pendentifsBpelleB	orientaleBorgiesBontilsBobs?d?eBnimbeBm?ratBm?moiresBmyBmultipliantB	mannequinB
mahom?taneBl?sineBl?chezB
luniverselBlop?raB	librementB
lexcellentBleucateB	lentendreBlascifB	langoisseBlaff?tBiris?sBinsigneBiniquesBimpr?gn?BillumineBguitaresBgibierBf?vrierBfroideurB
frissonnezBfresqueB	fourmilleB	fontainesB	fauteuilsBfaiblesB	enveloppeB
entretiensB	encha?n?sB	enchant?sB
effrayanteBeffondr?Bd?roulementBd?rangerB	d?laiss?eBd?fendBdiversBdisiezBdessineBdenvierBdenvieBdennuisBdavrilBdardeBcr?oleBcroyonsBcraintB
confessionB	composentBclownBcigalesBcharmaitBchacalBcertainsBcavesBcanonsBb?tirBblesseB	avonsnousBavionsBassassinB	arcencielBarbrisseauxBapollonBaccord?B
abominableB	?panchantB?cl?tB?cloreB?blouitB?avitriBvolumeBvivaisBversesB	verniss?sBursuleB	troupeauxBtroupeBtriangleB	treillageBtordusBtimploreBtendaitBtaillisBs?treB
s?veillentB
s?teignentB	survenantBsudBsouvraitBsilencieusementBsentinellesBsenivrerBsabbatsBr?veilleBr?jouirB	rugissantBrougiesBrobustesBrimeursB
resplendirB	renvers?eBramperBproiesBproc?sBpri?reBploieBplieBpeureuxBpeinesB
patriarcheB	orphelinsBonderBnulsBniaisBm?connueBm?chesB	minaudantBmeurtrirBmesdamesBmarmiteBlunesB	lumineuseBloursB	lornementBli?BlimplacableBle?tBlesquelsBlaureBlarvesBlarchetBlagateBj?touffeB
jentendaisBjaspireBjasentBin?galesB
inassouvieBimpiesB	illusionsB
hypocrisieBhoraceBhenryBg?niesBgreniersBglapitBf?rocesBfurieBfr?n?sieBfrapp?BfourmillantsBfolletBfatuit?BfaneBfBennuyeuxB	enfantineBelvireBeffr?n?sBd?pla?tBd?liresBd?cr?pitudeB
dossementsBdonnaBdisparusBdevienneBdevenuBdevaientBdestructeursB
descendantBdassezBc?lim?neBcribleBcreuserBcreusentBcrapaudsBcoureursBcordesBcoqsBcontorsionsBcom?teB
communiqueBcitoyenBch?risBchimisteBchass?BcatinsBblottiBbleuirBberceurBbaumeBbancBavironsBaveuBatelleBassautsBardeursBapr?sdemainBapprenezBangleBaim?mesBabelBabeillesBabb?B?voqu?B
?tonnementB?toil?esB?pelantB	?panouiesB?mulesB?chosB?maisBvol?BvioletsBvergersBvenueBt?n?bresB	trouverasBtrouvaisB
triomphauxBtra?n?esB	travaill?BtrahiBtournonsBtouch?eBs?par?sB	s?panouitBs?leverB
suppliantsB
subitementBsouterrainesBsiensBsem?sBsemblantBsedBsautBsatiataBsalueB
saintdenisB
r?bellionsBrocailleB	rencontraBremont?B	rembrandtBrejeterBrajeuniBquentreBpuisquenB	projetantBpoursuivantB	poudreuseBpotsBporteursBpiq?reBpercentBpa?enneBnonchalanceBnommaitBnatteBm?prisBmuseauBmortelleBmontaisBmontaiBmontaBmoindresBmissionB
merveillesBmarqu?eBmagieBl?cueilB	lorsquilsBlivr?eBlierBliensBlhumideBlendroitBlanimalBlamiBjuronsBjuiveBjetaientBissuesB
indulgenteB	incl?mentB	imprudentB	ignominieBh?bergeB
h?tonsnousBhoratioBhippogriffeBguetB	goutti?reBglac?esBgivreBgazouillantBfrappantBfranchementBfl?tesBfestonsBfangeuseBfameuxBexasp?r?BestampesBerrantesB	entrouvreBentrouvertsBd?sinvoltureB	d?sesp?r?Bd?nergieBd?livr?sBd?livreBd?faitBdissoutBdimancheBdignesBdiad?meB
descendonsB	dattendreBcurieuseBcr?veBcourstuBcoupaitBconsol?BconnusBconnaissezvousB
coccinelleBclosesBchienneBcharm?Bchang?sBcaress?eBcamaradeBb?tiBbruniBbrillerBboiteuseBblanchisBbisesB	battementB	banni?resBbanditBa?eulB	avidementBaventureBatti?diBattard?BatomeBarcadesBaquilonBam?neBameB	ambroisieBalliezBadmireB	accourentBabsenceB?touff?eB?poquesB?musB?l?ganteB?cumeB?claterB	?chevel?eB?claireB?maBv?n?reBvoudraBvitrailB	virginaleBveineBvailleBus?eBunirBtypeBtr?bucheB	trouverezBtrenteB	tremblaitBtorrideB	tordaientBtienB	th?ologieBs?pancheBs?duitBsuiventBsucc?deBspectralBsouplesB
souffriraiB	souffrantBsinueuxBsilhouettesBserr?sBsentezB	senivrentBseffa?aientBsecou?BscintillentBsaistuBsaisiB	sadossaitB	r?fl?chirBrusseBrugissementBre?oisBremousBrelev?BrasantBp?cheurB	prudhommeBpromusBprompteBprincesBpoissonBplongezBplanaitBphiltresB	pendantesBpaulB	panth?onsBpacteBorangeBopiumBombragesBn?tesBnierBnavoirBm?dusesBmourraBmort?BmorfonduBmonstrueusesB	mendiantsBmemporteBmatelotsBmangeaitBmalsaineB	l?ventailBl?onBl?ch?BlicornesBlarceauBlaidesBlaideB	j?taleraiBjuifsBjaspeBjalousieB	jaimeraisBjadoreBirionsBingratsBignor?sBh?tesB	hypocriteBheurt?Bg?teBguerrierBgougeBgageBfuseauxBfraternellesBfrancBfouilleB	fornarinaBflottantBfiletsBfermantBfavoriteBfatBfangeuxBesp?reBerranteB	emp?cheraBemporterBemb?cheBd?vor?sBd?troitBd?tacheBd?guiseB	d?coupentBduquelBdorantBdiraijeB	dincendieBdespaceBdentrerBdemijourB
dapplaudirBcrabeB	couchetoiBcorneBconfondBconflitBcommuni?BcoffreB	clairi?reB	chasseursBcartesBb?tisBbureBbr?legueuleBbonnetBbilletBberceuseB	belz?buthBbassetsBbarbaresBbalancesBbaisseBbabilBarchesBann?eB
animalpourBagneauxB	affaiblieB?toffeB?blouisBxxviiiBvaqueBusageBt?tonnementsBtyranBtr?sbeauB
trentedeuxBtousseB
tournoyaitBtonneBtent?BtarieBtaime?B	s?teindreB
s?raphiqueBs?mentBsuivonsBsuivisB	solennit?BsciencesBsavonsB	saper?oitB
salamandreB
sablonneuxBr?lesBruelleBroideBrevenuBrestetilBrestentBrenaissanceBrem?deB
remplacentBravageBrappelerBraillentB
quoiquelleBquinzeBquastuBpuresB	puissionsB
pr?destin?BpromptsBprojetsBprismeBprendraientB
poussi?resBpoursuisBplupartBplumageBploy?eBpli?BpimpantBperc?BpenchaitB
penchaientBpartisB	parall?leBpaphosBpaonB	onduleuseBolympeBobsc?neBnoblesseBnetsBnaufrag?BnationsBnagu?reBm?l?Bm?couteBmusicienBmultipliaitB	mis?rableBmissivesBmietteBmenantBmani?reBmangezB	l?l?ganceBl?g?resBlyB	luisantesBlucideBlondulationBloireBlogiqueB
lobscurit?Blhonn?teBlemporteBlavaientBlall?eBlaimantB
labsurdit?BjudasBjolieBjoliBjob?iraiBjetaitBjaseB	inventeurB
inqui?tudeB	incertainB
impassibleB	immensit?Bh?siteBh?b?t?BhumeBharpiesBharmonieB
guerri?resB	grotesqueBgrecqueBgrabatsBfussentBflorentinesBfeuill?eBfavoris?Bfan?esB	entendantBenfancesBencadreB	empourpr?BemployerBembusqu?B	d?routaitBd?mailBd?it?Bd?duitB	d?combresBd?choBdomaineBdogmesBditonBditalieBdisonsB
disiezvousB
dinnocenceB	deuxm?mesB	destin?esB	dendormirB
dapr?smidiBc?l?brerBcynthiaBcyb?leBcuistreBcr?antBcroulantBcraindreBcoulaitBcouch?eB
contresensBcontemplonsBcondamn?Bcoiff?Bclou?BclefsBchiffonnierBchassonsBchakalsBcasBcaptiveB
capitainesBb?n?dictionBbuterBbuisBbrickBbrefBbienheureuseBbercerBbayad?reB
basiliquesBbaraquesBbandesBbalconsBautomnalBathl?teBarmeBaraign?eB
approchonsB
apparitionBanguleuxBampleB
allongeaitBalbumB	adorablesB	accroupieBabsentesB?touff?sB?quipageB?puis?eB?plor?eB	?nervanteB?caillesB?coliersB?bonjourBvenaientBtuniquesBtr?veB
trouveriezBtrempeBtra?nesBtraduireBtir?sBtigeBtentezBtach?BsuceraiBsouverainesBsourdesB	soufflantBsommetsBsolideBserviBsecondBsaveursB	sapprocheB
saccouplerBr?volt?BrolandBrisibleBriraBrieuseBrepritBrepasserBreniantB
recueillirBrancuneBrafra?chissaientBquadoraientBp?gaseBp?sentBp?lerinsB
pulcinellaBpr?cisB
pr?c?dentsB
promenadesBpriantBprenaitBpoixBpiocheBpincesB
personnageBpenchaisBpasteursBparureB
parcourantBparadeBpantoumBpaletteBorni?resBondoyaitBoisiveB	n?penth?sBn?coutezB	nouveaun?Bnomm?Bna?adeBnationBm?lancoliquementBmusesBmos?BmoiresBmobBmaussadeBmaimezBlorgnonB
lorchestreBlinsondableBlexamenBlesquifB	lel?ganceBlattenteBlanguirBlampleurBjentreBirr?sistibleB	indicibleBincontestableB
incertaineB	imp?rieuxB	h?r?tiqueBhavaneB
grotesquesBgrisesBgrappesB	gracieuseBgalanteBgabarreBf?tceBfuseauBfrapperBfouettezBfoss?BforcerBfolletsBfacesBextr?meB	expliquerBexauceBespacesB	envelopp?BenvahitBentrouverteBentrentB
entra?nantBenterr?Benivr?BenchantementBd?sastreB
d?licieuseBd?chiresBd?chiffrantBdutBdupesBdonnonsBdompteBdolentBdocteBdiscr?tementBdialogueB	demandaitBdapprofondirB	dangoisseB	damertumeBcuissesBcr?tesBcriardBcourbantBcorbillardsBcoqBconqu?teBcomprendstuBcollerettesBch?tiveBchoeursBchemisesBchemineBceriseBcauserieBb?lierBbourdonnementB	bohemiensBblottisBbless?eB	becqueterBbaissantBbaignerBa?n?eBauteurBattiseBattaqueB
atrocementBar?meBall?goriqueBalcideBagilesBabondentBabattuesB?pouvantableB?plor?B?nigmeB?logeB?closesB?veB?o?BvolcansBvoisinBvoil??BvitresBvisitantBverresBveillaitBvanit?BtyrBtu?BtribuBtreillesBtirentBtigesBtenaisBtapiBs?rieuseBs?panouissaitB	s?culaireBs?blouitBsuivaisBsombr?B	sinfiltreBsib?rieBservanteBsenroueBsavoureBsautentBsattacheBsanzioBsalomonBsaffligeBr?v?eBr?vezBr?volutionsB
r?volutionBrousseBroseauBronceBrivesBresteraBressouvenirBrespectBremplac?BremettreBreliefBrayonnaientBraviveBraillaBraideBquittantBp?tritBp?m?eBpunaisesBpr?par?Bpr?jug?sBprudentsBprudentBprouverBproph?tiqueBprojetBprofitBproduitB
prisonnierBprenezBpouxBpoulsBpharesBpeureuseB	passerontBparosBouvrirBordresBoffenseeBnoirciB	neigeusesB	naissanceB
m?chancet?Bm?choireB	mysticit?B	murmurentBmouetteBmeurtritB	mendianteB	mauvaisesBmatinsBmarchesBloyalBlorni?reBlorgieB
linvisibleBlholocausteBlesclaveB
lempreinteBlathosBlardBlanneauBlaimerBjupesBjolisBjetezBjacobinBinscriptionBinfiniesB	incomprisBincivilBh?ritageBh?tantB
hydropiqueBhunsBhaieBgrondaitBgriveBgraineB	gourmandeBglacesBgaresBgagn?Bf?ch?BfrissonnentB	fouettantBfondsBflattentBferontBfatigu?eBfascinerBenvironnantBeffraieBd?vieBd?tailBd?pouxBd?cusBd?chir?B
d?chirantsBd?battreBdoublierBdonnantBdocilesBdh?l?nusBdevinsB	derni?resBdensesB
demprunterB	danarchieBdalleBdadieuB	c?l?brantBculotteBcuirBcueilliB
creuserontBcreusantBcraintifBcoup?BcoquilleB	coquettesBcon?uBconsumerontB	conqu?rirBcoiffureBclairobscurB
cimeti?resBchrysalidesBcheveluBchacuneB
cendrillonBcatonBcarcansBcapricesBcapeBcamailBb?r?niceBb?illonsBbuffetBbrocartBbouclierBbagueBauxquelsB	aust?rit?BaudevantBatoursBath?esBassouvirBarm?BardentesBappauvriBampleurBamaigrieBallumentBallong?sBallaisBaidantBaccourtBabsentB
?v?nementsB?pargn?sB?normit?B	?namour?eB?couterB?ilsBxxviiBv?h?mentBvirentBviolentsBvestaleBvernisB	vengeanceBvendeursBvatteauBtr?nerB	tr?pass?eBtrouvezBtramentB
trafiquantBtrac?BtoujourscommeBtouffesBtortuBtitansB	th?roigneBteint?sBtapaiserB
s?parpilleBsuperbesBsuivezB	souvenaitBsonnezBsoirl?Bsi?geBsirriteBsinclineBsicileB	sepultureBsentitB	senivrantBsemerBselBsculpt?eB	sanglanteB	salutaireBsaharaBr?dantBr?serveBr?gnasBr?duireBromptBriensB
reviendrasB	retournerB
respectentBrepaireBrelevaitBregrett?BrefluantB	redoubl?sB	reconnaisBraviB
ratatin?esBrapproch?esBrappellerasBrallumeBradieuseBp?n?tr?B
p?n?trantsB
p?lerinageBp?t?sBputainsBpugetBpr?tesBpr?sideBprudeB	promessesB
prochainesBpoursuivonsBposezBpoitrailBpoeteBpleur?BplauteB	phtisiqueBperdaitB	parisiensBpalmierBpaisiblementB	pacifiqueBorbitesB	ondoyantsBombrag?BoeuvresBoeufsBodorantBobscurciBobiBn?galeBnymphesBnuquesBnovembreBnou?sBnouvrantB
nourrissonBni?BnicherB
nentendantBneigerBnavaientBnaufrageBnadirBm?laitBm?lodieBm?lang?Bm?nentBmousselinesB	mortalit?BmoraleBmoissonBmiauleBmexercerBmesurantBmessieBmaladesBmachinesBl?t?BluBlortieBli?vreBlisaitBlh?b?t?BlhydreBletheBlarveB	lanath?meBlam?sB	lalouetteBlalbumBlaharpeBlabsenceBjunisBjourdainBjerraisBjaillisBiraitBinvuln?rableBinjusteBing?nuesB	infortuneB
indulgentsBindolentBimpuissantsBimmens?mentBhuneBhumeurBhoulesBhostileB	honteusesB	habitantsBgu?rieB
grincementBgrimpentBgriffonsB
grelottantBgardaitBgaleriesBgaillardementB	furibondeBfrottantBfraudeBfrascatiB	fran?aiseBfranchiBfragmentBfoireBfl?trirBfluideB	floraisonBfid?lit?BferezBfavartBfatigu?sBexercerB
excellenceB	etonnantsBessorsB	esclavageBepriseBentoureB	ensevelisB	engendrerBenfum?sBd?vorantBd?tailsBd?roul?B	d?pouilleB	d?passantBd?licieusementBd?gagentBd?crocheBdragonBdouzeB
divinementBdinconnuBdhectorBdexisterBdespoirBdellesB	delacroixBcr?duleBcroupBcrispantBcrapuleBcoursesBcourb?eBcoteauxBcort?geB
convulsantB	contrist?BconserveBconscritB	conqu?tesBconnaissaitB
confidenceBconfess?BconduireBcompterBcompasB	comparantBcommenteBcolombBclameursBcirconstancesBch?rirasBchevelusBcentupleBcarnavalB
cariatidesBb?tonsB
bouvreuilsB
bouteillesBbondisBboeufsBbatteuxBbasaltiquesBbaiss?sBbaign?B	attirantsBattentifBattelageBath?eBasthmatiqueB	apportaitBapaiserB
anachor?teB	amaryllisBaltierBallezvousenBallaitaBajustantBaimezmoiBagitentBadorablementBadelineB	accueilleB1839B1838B?tuiB?toffesB?pelerB?croulementB	?crivainsB
?clatantesB?blouissementsB?unBvousm?meBvol?esB	vignettesBverguesBvautreBunesB	troublentB
tristessesBtravaill?cestBtrapusBtransperc?eBtrag?dieB	tourellesBtomb?sBti?desBtirezBtinfuserBtictacBthalieBtestBterrainsBtenl?veB
teignaientBtalonB	s?r?nadesBs?lancerB
surchargeaBsultanBsuffireBspinosaBsoulageBsortesB	songeatilBsompteuxB
sinclinentBserrerBsembrouilleB
seffeuilleBsecourirBsaulaieBsatin?B
sanglotantBsaimentBsaccrochentBr?derBr?tiaireBr?gionBronsardBriezBrieurBricardBrevenueBressusciteraientB	ressortirB	renvers?sBrendsf?eBremetB	religionsBrelentBreinesB	refermentB
redresseurBrebuteBramasserB	racontantBquhabiteB	pythagoreBpyrrhusBputBpuissentB	pr?f?ronsBpr?f?reraitB
proserpineBpromen?BproduitsBpreuveBpouvantBpourronsBpourpresBpositifBpontifeBpolymnieBpluiesBploy?BpleurezBpinceauxBpiloteB
pierreriesBpersisteB	permanentB	peinturesBpavanerBpastelsBpartantB	pardonnerB	parcellesB	ouvraientBorphelinBnympheBnuptialBnudit?sB	nousm?mesBnouantBnommesBnentendeB	naufrag?sBm?lezBm?nesB
mousselineBmourraisBmopprimeBmont?sB	monarchieBmoineauxBmodesBmis?rablementBminuitsBmimoirB
meurtri?reBmettraBmetamorphosesBmentaitBmasureBmassiedsB	marquisesBmanquaisB
maladroitsBmaladifBl?vrierBl?trangeB	l?normit?B	l?l?phantBl?veraBlyresBlupanarsBlouvreBlouvrageBlombrageBlisonsBlindulgenceB	limprimerB	limmortelBlimeB	lignorantBliaitBlev?eBlevaientBlesp?ranceuneBla?eulBlauteurBlautanBlaust?reB	larchangeBlaper?oistuBlanguissantsBlanguissantesBlam?esBlaissesBjuponBjobBjet?esBiris?eBireB
inventeursBinerteBimpursBimitonsB
imb?ciledeBhochetBharmonieuseBguiseBgrossierB	grimoiresB	gratitudeBgaspardBgardesB	galammentBf?l?esBfunestesBfum?BfrissonnementsB	fredonnesB
fraternit?BfranchitBfoyersBfonteBflamboiementBfilentBfesseBfaistuBfaim?BexplicationBeuterpeBeurentBeunuquesBetoilentBeternelBenrhum?eBenlac?eBenivrantB	engendrasBenfum?eB
empress?esBemmitoufletoiBeffac?sBd?terrerBd?ploieBd?pens?eBd?livraientBd?faiteBdurablesBdoguesBdodeurB	dendymionBdardantBdamneBc?bleBcuistresBcr?ateurBcrinB	craintiveBcouvesBcoulisseB	cothurnesBcorrecteBconviensB
continuantBconquisBconfierBcomptaiBcompl?teBcomplaisammentBcomparerB	commenc?eB
claquementBclairvoyantBcitronBciterneBciresBcirconspectionBch?tiantB	chuchot?eBcausantB	cantiquesBb?nisBbr?lentBbrinsBboutonBbourdonsBboucl?sBbordureBblaseB	blafardesBbisBbienfaitBbassinsBbananesBbanaleBballadesBballadeB	balancierBbalancementBbaiss?eBautansBaurionsBauquelBattraitBattacheBatelierBarracherBarchitecturesBapporteB
anterieureBanglesBananasBagit?B	adorationBaccrueBaccruBaccabl?eBabrutissantB0B?lotB?vapor?eB?touff?B?teindraB?talonB?puiserB?priseB?pancheB?nervantB?lixirB?glisesB?gayerB	?galeraitB?gaieB?cussonB?cumantB?cueilsB?cueilB?clips?B?clairciB?chafaudagesB?voyezB?uneB?questceBz?phirBz?leBweberB	v?n?rableBvoudrasBvoleursBvoil?BvivaientBvitrageBvigieBviennesBvertigineuseBverdierBvassaleBvarieBurnesBt?claireBt?tonsBtumulteB	troism?tsB
triomphonsB	trembloteBtra?tresBtransparenteBtoutpuissantsB
tournoyantB
tourmentezBtortilleBtorduesBtoqueBtocsinBtivoliBtintentBtigresseB	thermodonBtessonsBtacheB
tabernacleBs?resB	s?puisentBs?meraB
suspendentBsusBsurpasseBstatureB	soupirauxB	souffrentB	souffleurB	soucieuseB
solitairesBsodomeBsinclinaientBsimo?sBsillageBsignal?BsierraBservilesBservageBsavan?aientBsaugeB	sataniqueB	salourditBsainBsaign?Br?leBr?v?sBr?vuls?sBr?jouisBrugirBrousseurBros?treBrongezBroidesBretientBrestesBressemblantBreptilesBreluiseBregretteBregagneB	redresserBrecuitB	recueilleB
recommenceBrechigneBrebelleB
rayonnantsBravag?BraresBrappelleBrameursBraisBracorniB	quic?taitB	querellesB
quensembleBp?tulantB	p?tillantBp?nombreBputr?factionBpucelleBpr?t?B	provincesBprosodieBprolongeBprofanesB
proclamantBpriv?BpressezvousBpo?mesBpoulesBpouacreB	portaientBpopulaceBpluvi?seBpliantBplatesB
planterontBplanantBplac?sBpi?ceB
pieusementBpeupl?sB
persiennesBpaysansBpavotsB
passezvousB	participeBparlaisB	pantelantBpalperBopaqueBombrageaBoffertBoeilladeBodoranteBobserv?BneuvesBnentendsBnemrodBnauraiB	naturelleBm?laiBm?cr?antBmoutonBmousBmoqueursBmontreznousBmoitesB	militaireBmiclosesBmettraisBmenveloppaitehB	menterrerB	memporterB	maternelsBmatelotBmatBmarchezBmaraudeB
mapprendreBmal?dictionsBmaireBmadmireBl?tendueBl?preuxBl?lixirBl?chet?Bl?cheraBluxureBlouisxavierB	lopprobreBlitt?ralementBlisolantBlirreparableBlimpuissanteB	lidol?treBlidealBlhosannaB	lheureuseBlhabitBletreBlenteurBlennuiloeilB
lembrassesB	lembrasseBlantiopeB	lamoureuxBlamentoBjouesBjaunisB
jarreti?reBjapporteBjambonBivrognesBinspirerB
insoucieuxBindiff?rentsBinclinerBimmortellesBimberbeBignoraitBhumilit?BhoukaBhautboisBhasardsBhameauxB	g?om?trieBg?n?reuxB	guerri?reBgr?ceBgroup?sB	grondantsBgrillesBgravierBgoyaBgouverneBgoudronBglasBgiguesB	gaspillerBgardantBf?condsB
fussentilsBfumeurBfuieB	fr?quent?BfrontonB	fouillentBfortunesBformuleBformidablesBfondusBfleurisBflandreBfinalB	figuretoiB	fauvettesBfaussetBfamillesBfalloirBfadaiseBescrimeB
escaladantB	envieusesBentablementB	ensevelirB	engloutisBenfl?Benfant?B	endormiesB	endolorieBencha?n?Bemprunt?B
emprisonn?BembarqueronsB
d?votementB	d?voilentBd?truitBd?thersBd?shabillerB	d?salt?r?Bd?rob?eB	d?racin?sBd?pos?sBd?noueBd?gliseB
d?daigneuxBd?combreB
d?chafaudsBdusageBdressaitBdoutezBdoutentBdoubl?sBdissipeBdiscordsBdinvinciblesBdintimeBdiamant?BdhermineBdexilBdexhalaisonsB	destampesBdespoteBdescendsB	denchant?B	darlequinBdardsB
dam?riquesBdallesBdaffaireBdacheterBdabyssienneBdabandonnerBc?l?breBc?linsBcupidonBcr?mentB	croyezmoiB
crepusculeBcouveBcouch?sBcouch?esB	convuls?sBcontoursB	constanceBconspirateursBconqueB
conna?trasBconfessionnalBcompos?B
complimentB	commencerBcoloreBcocoBclasseBcitadinBciseleurBcimeBchutesBchoisiesBchenilleB	chemin?esB	chatimentBcharmezBcharlesB
charitableBchangeantesB	calembourBcachaB	b?nissonsBbuveursBbuvaitBbulleBbrutusBbrisantBboxeurBboug?BbosseB
bercementsBbazarB
basreliefsB	basiliqueBballonsBbaign?sB
babillardsBavari?sB	avaleraitB	avalancheB	aurezvousBauratilBattach?sBarm?eBargentinBappuy?sBappelerB
apparencesB	all?chantBalchimistesBaiglonsBadoptifBadmirerBaccourezBabriteB?taitB	?touff?esB?tonnementsB?tirantB?p?leB?pousesB?pisB?meraudeB
?lectriqueB?coeureB?bauch?sB?surB?quiBv?tueBvrilleBvoientBvoguentBviveursBvillagesBvilaineB	vibrantesBvauxBvaurienBusurperBt?norBtutelleBtroublesB	trottoirsBtriturerBtrism?gisteBtressaillantBtremp?sBtra?naisBtraqu?BtournureB	touchanteBtircisBth?ologalesBth?B	terrasserBtendusBtenduesBtenaientBtempsl?B
tembrasserB	tattaquerBtamarinsBs?tireB	s?pultureBs?puiseBs?dentairesB
s?dentaireBs?criaBs?claireBs?chantB	s?cartentBsyllabeBsueursBsuairB
splendidesBso?leraiBso?lerBsoyeusesBsoudainsBsotteBsortonsBsorcierBsonneraBsongeursB	sommeilleB
sobscurcitBsil?neBsilencesBsexileBserviraBserrureBsenhardissantB	senfuyaitBsendormaientBsemantB	semaillesBseffrayaBschismeBscapulairesBsauverBsatyresB	sasseyaitBsarr?taBsaphirsB
sanglantesBsal?sBsafranBsadaptaientB
sabreuventBr?vaisB	r?veillerB	r?servoirBr?p?tantBrutilantBroyalesBrou?sBroulaB
rougegorgeBrongeraBrompusBroideurBrientB	richessesBrhinB	revinrentB	resteraitBrepueB	reprenaitBrepara?tBrent?BrentronsBrendezBrena?trontilsBremonteBreluisaientBrefroidiB	redingoteBrec?lentB
recouvertsBrecelantBrappelezvousBquyBquoiquilB
quitt?rentB	quinquetsBquartsBqualorsBp?niblementB
p?dagoguesBp?cheursBp?mentBpr?taitBprostitutionB	proscritsBprofilsBprisonniersBprenneBpoursuivraitBpoursuiviesBpoteauxB
possessionBpossedeBporeuseBpolypeBpleuresBplanterBpitreBpiquesBpindareBphilosophesBper?antsBperdraitB
patriarcalB	past?quesBpassagerBparterreBpars?meBpaniqueBpailletaBornerBn?eBnuitfugitiveBniveauBnardBnageraB	m?prisantBm?ditantBm?tB	multipli?Bmoi?Bmod?lesBminviteB
mintimidasBmimeB
mhumiliaitBmeubleBmentreBmentirB	menivrentBmenaceBmaudisBmatinalBmasseBmalsainsBmaladiveBmageBmademoiselleBl?pineBl?onardBl?go?smeBl?gendeBl?blouissementBl?creB	lovelacesBlouveB	loutremerB
lourdementBlobjetBloasisBlivoireBliseronsBlinfirmeBlimpuissantBlimmortalit?Blh?tesseBler?beBlenivrementB	lempereurBlavenueBlauventBlardentBlarcherBlall?gresseBlalg?breBlaisseznousBlaisseraBlaineBlaimaisBlaffrontBladyBlabyrinthesBjulietteBjug?eBjadmireBirrit?sB	in?puis?sBintervallesBinsens?sBinsensiblesBinondentBinnombrablesB	infid?lesB
impudencesBimm?diatementB
illuminaitBid?alBicarieBh?pitauxBhurlantsBhueBhouilleBhotteBhorizontaleBhommagesBherm?sBha?rBhautainB	hardimentB
gu?risseurB	grognantsBgourdeBgonflerBglabreB
girouettesBganguesBgalopsBf?tsBf?cond?BfutilesBfum?esB	fouettaitBfortifieBfontainebleauB
flamboientBflambaitBflairantBfiltrerBfilousBfileuseBfeuilletantBfertileBfeleeBfarceBfaquinBexil?Bexerc?eBentrantBentraB	entendrezBenormeBennoblitBenivrerBencenseBencadr?BelectreBeademBd?votesB
d?tranchesBd?toursBd?sordonn?esBd?sir?Bd?pass?sBd?normesBd?mesur?B	d?lectionBd?go?tsBduvet?sBdormiraiBdormiBdorentBdorduresBdommageBdixseptBdivansBdispenseBdisloqu?BdineffablesBdimp?rissablesB	dimmensesBdevraitBdevenusBdespagneBdescendaientBdenvelopperBdenseBdeminueBdathl?teBdarmesBdandysBc?toyerBcr?teBcroyanceBcroisseBcrisp?eBcreusetsBcreoleBcrat?reBcousineBcoupletsBcorailB	copierontB	convulsifBcontempteursB
contemplesBconsum?sBconstantBconsommeB	consacr?sB
confonduesB	confianceB
comprenaisB	combattezBcoll?eBcloueB
clochettesB	clepsydreB	chouettesBchevaletB
chercheursBcheminerBchauffezBcharlemagneB
chapiteauxBchapeletBchantaB	changeantBcaveBcavalesBcausonsB
campagnardBcalliopeBcafrineBbuscBbruineBbrouteBbo?teB	bourdonneBbord?BboitantBbistreBbercentdautresBbenedictionB	battirentBbattantsBa?euleBautruiBaurezBaudaceBatteinteBatonieBatlasBassoupisB
arbrisseauB	apportentBambianteB
allumaientBallionsB
alimentonsBalchimieBalarmeBaimionsBaimaientBagn?sBagilit?BadolescentsBadmirantBactifBaboutitB1830B?vanouirB?tincelantsB?l?phantB?dificesB?crinB?carlateB?blouiesB?prisB?denB?nousBwagonsBvousceBvolutesBvolantsBvoisineBviviezBvisibleB	vingtneufBvindansBvinciB
viendrastuBvid?eBvibraitBviandesBveuleBvertueuxBverditBverdisB	venezvousBvelout?sBvaugelasBvaporis?BusentBuniquesB
uniquementBt?moinsB
tyranniqueBtu?eBtutBtrouvonsB
trouverontB	troublantBtroph?eBtraduitBtourterelleBtournoisBtourelleB	toilettesBtiroirsBthomasBterneBtermineB	tentationBtaquinBtaistoiBtainBtablierBs?tendB
s?pandrontBs?mousseBs?antB
suspendantBsursautBsurfaceBstupeurBstrygesBsto?quesBsparteBsouvienstoirapideBsoupirerBsoumetsBsoulagementBsouill?eBsoudainementBsondesB
sintroduitBsinterromptBseuletteB	senvolantBsentimentaleBsemperBsemblablemonBsceptreBscandaleB
sauterelleB
sataniquesBsapprivoiseBsamuserBsaccadesBr?vonsBr?volutionnaireBr?veill?Br?p?t?B
r?pugnantsB	r?pandantBr?jouissaitB	r?alisaitBr?telierB	r?lementsBruin?sBrivaleB
reviendraiBretenuBretentissantesBrepusBreprendsBrenvoy?BrendentBremuerBremuentBrememberB	regrettezB
regrettantB	redressesBredorerBrecueillementBrecenseBrayonnerBravinesBravaleBrajeunisBraffinementsB
qu?crasentBquinfiniBquaujourdhuiBquattir?BquatrevingtneufBquaiB
p?trissantBp?risBp?rirBp?rilB	p?nitentsBp?querettesBpyladesBputrideBpuentBpr?tendBpr?senceBprovocateursB
promettantBprisesBpo?meBpoup?eBposticheBposeraiBpomperBpomoneB	plongeraiBpleuretelleBplatsBplateB	plastiqueB	plaintiveB	plaignentBpirouettantBphiltredansBperspectiveBperch?sBpensonsB	penchantsBpa?treBpassanteBpascalBpar?eBpartionsBpars?mesBparqu?Bparfum?eBparaisBpalpableBouvrag?sBorfraieBorduresB	orchestreBonduleuxBoccupentBocculteB	ob?issantB	obscurcisBobliquesBn?gligeB	n?cessit?Bnubilit?BnoyezBnousm?meBnourrissentB
nourricierBnontilsBnoieronsnousBnoffusquaitBnoffenseBnichesBm?treBm?riteBm?fianceBm?nemoiBmusqu?sBmouvantB
moutonnantB	mourantesBmoueBmoscouBmordentBmordeBmollesseBmodeBmiroiterBmeusB
meurtriersBmeurtreB	mennu?raiBmend?sBmauresB
maternelleBmaternelB	massistesB
malfaisantBmagisterBmagicienBl?troitB
l?pouvanteBl?pieuBl?crinB	l?cheveauBl?cherBl?cherB	loeilladeB
lobserventBlitaniesB
linsolenceBlienBleuropeB	leternit?B	lencolureB
lemp?chentBlembl?meBleffraieB
lecteurmonBla?smonstresBlav?BlavertisseurBlattraitB
lancinantsBlambrisBlamasBlall?eB	lalbatrosBlabhorr?BlabandonBj?prouveBjumellesB
jentrevoisBjarriveB
jabandonneBirriteBintimesBintelligenceBinsolemmentBinscriteB	infernauxBinfecteBinfamieBindomptableB	inapais?sBinaccessiblesBimposerBimplacablesBimiteBid?aleBh?leB
hypocritesBhumeursBhorreursBhistrionB
hirondelleBhilaresB	harmoniesBhamletBhaloBhaiesBg?nuflexionsBg?henneBg?teauBgrossesB
grassementBgouleBgorg?sB	gloussantBglapissantsBgiffleBgazetierBfuturB
fuligineuxBfr?lesBfrugaleBfrayeBfran?oislesbasbleusBfourreauBfourmiBfluetsB
flottantesBflasquesBferm?eBfa?tesBfatiguesBfatalesBfanauxB
fam?liquesBfameuseBexpierBexhalantBeponineB
envelopperB
entretientB	entonnoirBengendreBendroitBendormeusesB
endormeuseBencombr?B
encha?nentB	enchant?eB
empourpr?sBempest?sBembl?mesBeffray?Bd?vou?sB	d?tachantBd?sireB	d?sirableBd?routeB	d?quipageBd?fitBd?fiBd?faitesB	d?cr?pitsBd?coleBd?clairsBd?chusBd?bauch?B
dressaientBdou?B	dopulenceBdoiseauBdisposB	disloqu?sBdinvisiblesBdinsecteBdinfiniB	dhorizonsBdevinonsB
devinaientBdestructeurB
descargotsBdeniseBdemoisellesB
demiclosesB	demandantB
daraign?esB
dapresmidiB	dansaientBdallumerBdabandonBc?sureBcuirasseBcr??Bcr?piteB
cr?pusculeBcrev?BcouvrentBcouverBcouraitBcouchezvousB	coucheraiBcontesB	construitB	condamn?eB	compteursB
comprendraBcomplotsB	complicesBcommen?aBcommencementBcomiqueBcoch?resBclairvoyantsBclairvoyanceBcitadinsBcirc?Bch?teauxBchuchoteBchercheusesBcheminezB	chassieuxBchargerBca?nBcavernesB	carillonsB	caressaitBcapoueBcB	b?cheronsB	b?quillesBbustesBbrunitBbris?sBbris?eB	bric?bracB	breloquesBboulingrinsBbougerBbouffeBbossusBbosquetsBbilansBbelleauBbavardBbariol?BbalsBbaisionsBbaisantBayonsBavenirBauronsBattraitsBarsenauxBarqu?sB
aromatiqueBarmatureB	apr?smidiBappuiB	apparenceBantino?sBamortisBamiti?BamantesBalimentBalbatrosB	aimonslesBaideBagrandieBagiterBagentsBaffrontB	affreusesBadmisBab?meactionBabritaitB	abandonneB183B?veill?eB
?touffantsB?therB?taisjeB?tag?resB?paisseB?cussonsB?crireB?coutantB?oB?auBxxixBv?ron?seBv?g?taleB	vousm?mesB
voudraientBverraijeBverminesBvaillantB
vacillanteBus?sBusantBt?tantBtuilesBtuaBtrouv?reB	tropvoil?BtricherBtresserB
traversaisBtourmententBtomberaiB	tombaientBtissusB	th?ram?neB	texprimerB
testamentsBternisBtenvieBtendueBtaimerB	tadmirentBtadmireB	s?vaporerBs?vaporeBs?ch?eB
surlechampBsuffitilB	suffisantB	so?lerontBsouterraineBsoupirsplusB
souffrirezB	soufflonsB	soufflaitB	sonnettesB	songeriesB
somnambuleBsolidesBsoistuBsoffreBsocBsoaBsinstall?rentB	sillonnesBsergeB
sentimentsBsentimentalB	seffondreB	scrupulesBsavaitBsavaientBsauraitBsaufB	sassouvirB	sarr?tentBsallongeaitBsaignerBsadresseB	saccusentBr?deursBr?volt?sBr?ponseBr?pitBr?jouitBr?fl?chirontBr?cifBr?teauxB
ruisselaitBruineuxBrubanB
royalementBroulierBroucouleBrompantBricaneBrevisBrevintB	retrousseBretournaBretentirontBreste?BrestaitB
ressasieraBrespect?BrentrentBrentraitBremu?esBremerc?mentBremarqueBregardaBrefra?chiraBredouterBravinsB
rassemblerB	raquettesBrapporterasBrapporteBranceBrampentBraieB	raffermitBquorBquoiquonBquittesBquavrilBquautourBquassaisonneBp?tB	p?trarqueBpuret?BpulluleB
puissancesBpr?cocesB
pr?cipicesBprusseBprostitu?esBpriveBprisonni?reBprenonsB	prenaientBpoussaBpourvoirBpourrezBpourpr?sBpourpr?eBpomponsBpompeuxBpointusBpl?tr?eB
plongeronsB
plateformeBpi?cesBpirateBpesantsBpermetB	perditionBpendueBpayeB
passag?resBparliezBpardonnezmoiB
paraissaitB
pantouflesBpanneauxB
palaiseauxB
paillettesB	paillasseBoserBop?raB	opprimantBoctog?nairesBoboleBn?migronsnousBn?fastesBn?buleuxBnoy?sBnoy?esBnoyantBnouveaun?sansBnourriceB	noblementBniantBneigesB	naufragesBnargueBnaimiezBnaimerBm?tamorphosesBm?fiantB
m?daillonsBmutinsB
moustiquesBmorfondBmordezBmoquentBmiraculeuseB	messieursBmerveilleusesBmemorB	martyris?BmartinetBmani?r?sBmadorerBmachinalB	l?trangerBl?tourdissanteBl?touffeBl?tatB	l?ph?m?reBl?couterBl?veraiBlutinsBlunissonB	lubriquesB	lorphelinBlirr?missibleBlirremediableBliqueursBlinvitezBlimpalpableBlima?onsBlhomme?BlheautontimoroumenosBlevonsBlevaitBlessaiBlequelBlenviB
lendemainsBlasciveBlancreBlallureBlaimaiBlac?sBlabsentBjusquo?B
joueraientB	jescaladeBjavertisBjaimaisBjBirr?m?diableBinviterB	int?rieurB	intestinsBinstinctBinond?esBinitiumB	infortun?Binflig?B
incompl?teBimpr?vusB
impossibleBimpitoyableBimpieB
imparfaiteBimbibeBillumin?BicareBh?ro?smeB
hurlementsBhorlogeBhideusesBha?ssantB
hainecommeBg?anteBgrandirastuB	grandioseBgrandiB	glisseraiBgla?antBgisBgavarniBgalantesBf?tesBfurtivementBfr?missaientBfrontonsBfrissonnanteB	frapperaiBfourmillanteBfossesBformantBforgesBfolBflueBflorenceBflamboyaBfilantBfermaitBfascineB	faisaistuBextravagantsBextravagantBextraireBevesBetresBestoBermiteBepiantBentrezBentrevueBentassementBenray?BennuieBenflamm?BenflammeB
enfantinesBencolureB	embaumentB	elevationBeffacentBecraserBeclosesBd?nersB	d?veloppeB	d?ternelsBd?lirerBd?go?t?Bd?cretBd?cor?Bd?bordaientB	d?battantBduelsBdress?BdoublesBdormirasBdoraitB	donnezmoiB
donnezleurBdominationsBdoeuvresBdinf?mesB	dimpr?vusB	dimplorerBdiminu?eBdh?pitalBdheureBde??BdessousB	descenditBdegr?BdeauxBdarchetB	dalouetteBdairB
dagonisantBdadieuxBcythereBcuivrentBcrutBcroisentB
craintivesBcoureurBcouranteBconvulsionsBcontenaientBcontactB
constell?sBconfinsB	confidentB	condamnerBcommandeBcolloqueBcolletBcogn?eBciviliseB
ch?timentsBchoisisBchellesBchaumineBchaumeB
chaudementB
chastementBchasseresseBcharonBchant?sBchaisesBcatinBcastagnettesB	carrouselB	capitalesB	capitaineBcapacit?BcalcineBb?titB
brodequinsBbrisaitBbravesBbravantBbourdonnaientBboudoirsBbosquetBbomb?sBboh?meBblouseB	bic?phaleBberg?resBbasilicBbapt?meB	banquiersBballetBbadinBbabilleBavou?BaventureuseB	aventuresBavaresBauroresBastrologuesBassourdissanteB	assourdieBasilesBarriv?B	arrachantB	araign?esBamphoreB
all?chantsBallumantB	allegorieBajoutentBagranditBagacentB	adressaitBactesBacierB	accumulerB
accrochentBabsurdeBabreuverB1855B?lotsB?tincelanteB?mauxB?lueB
?lastiquesB?glogueB?critsB?corceBz?lesBv?fourBvulgaireB
voltigeaitBvolonsBvoitureBvitonB	virulenceBversezBvermisseauxBveniseBvenantBvaudraitBvacheB	t?te?t?teBtyransB
tr?pignantBtr?piedBtrottentBtroncsBtrompeurBtromperBtromblonBtringleB
tremblanteBtouchantBtordraiB
tonnerastuB	tombeauceBtirsBtienstoiBth?oneBth?meBtermiteBtendormiraiBteintsBteintesBtappellerontB
tappellentB	s?prenantBs?largirB	s?goutterBs?cherBs?battreBsyst?mesBsupporteBsuffisammentBsubtilsB	stupidit?Bstigmatis?sBsouperB
souffrantsBsoublieB
songeaientB
sommeillerBsionBsingesBserrantB	serezvousB
sentinelleBsemeurBsecouaitB	scorpionsBsauraisB
satisfaireBsassiedB
sarrachaitBsappuyaBr?sinesB	r?fugiantB	r?clamaisBr?alit?sBr?volteBrythmesBrucheBrouill?Bris?eBridesBriboteBrevoleBrevenonsBretroussantBrestezBrestaur?Brepr?sentaiBrepentB
rendraientBrendraiBrenardBremu?eBrel?gu?BrefusaitBrefl?terBrefleuriBredoutablesBreditsBrateBranimerB	quoph?liaBquittezB
quittancesBquenflammaitB	quenfanteBquassouplitBquartBquamisBquaijeBp?nitentBpupitreBpuiserBpubliqueBprovoquantsBprouveBpromontoiresBprojetteB	projetaitBpriseBpresteBposthumeBport?Bpomponn?B
pleuvrastuB	plaisantsBpip?Bpes?B
pestilenceBperfidesBpensaBpa?ensBpa?enB	parfumantB
paracheverB
palpitanteBosseuxBosineBopulentBopini?trementBomnibusB	offusqu?sBnoy?eBnoyaBnommerBnativeBm?rirBm?langeBm?criaisBmytheBmufleBmuettesB	morfondueBmordrontBmoiteursBmobileBmivoixBmin?rauxBminesBmicorpsBme?tBmesurerB
mengloutitBmavoirBmavanceBmar?cageBmarqu?sBmarionnettesBmargelleB	maraudeurBmaraisBmanoirsBmanoirBmani?reBmagn?tiquesBmagesBlucarneBlubriqueB	lubricit?BlorgnantBlinsensibilit?BlinceulsB	lillimit?BlicesB
lhonn?tet?Blexc?sBlev?B	leum?nideB	lescabeauBlazzisBlavoineBlavidit?BlavesBlattaqueBlarri?resaisonB	larchipelBlancentBlamiti?BlaimableBjuraitBjumeauBjoseBjordonneBjet?eBjenlaceBjaserBivremortB	italienneBinviol?B	instinctsBinstall?B	insolubleBinondantBinfectsB	infectionBinfatigableB
ineffablesB	incr?duleBincorruptibleBinconstantsBimperceptibleBimmortelB
immensit?sB
illuminantB	hurlantesBhonneursBhommesl?BharpagonB	haranguesBhal?teBguivreBguitareBgrossitBgrosseurB
grima?anteBgreenBgrandgousierBgo?theBglorifierasBgla?eulBglanerBglaci?reBgentillesseBf?l?eBf?l?BfriseBfraterniserBfraisesBfourbuBfor?antBformulesBforg?BfondeBfi?vresB	finissentB	feuill?esBfemelleBfaonsBfan?sBfalluBfalaiseBfabliauBextirperBexploitsB	expliquesB	excessiveBexaudiBexactBeveilleBetoileBessayaitBesB	envoletoiBentreb?illantBensuiteBempoisonn?sB
emphatiqueB
d?salt?rerBd?rangeB
d?pith?tesBd?funtesBd?faillanteBd?charneBdurcitBducBdroitesBdovideBdornerBdonnemoiBditesmoiBdistinguaitB	discutantBdinsondableBdimpursB
dimpudiqueBdhelminthesBdestin?sBdemandaientBdautruiBdapparitionsBdanseuseB	damnationBdalorsBdalarmerBcultesB	cr?aturesB
croyezvousB
croulantesBcrois?esBcroieBcritiqueBcrispeBcrachatsBcorrompuB	coquillesBcopp?eBcontemplantBconsacreB
conqu?rantB
cong?n?resB
condamn?esB
compteraisB	compagnieBcolossalBcohorteB	cliquetisB
claviculesBclameB
citadellesBch?tr?B	chr?tiensBchoifBchevetsBcha?nonBchaumesBchasseurB	charriantB	charleroiB	chanteraiBchacalsB	causeriesBcartonB
caricatureBcaptifsBcanevasBcanauxBcampsBcampoBcabaretB	b?atitudeBbyronBbulletinBbr?l?sB	br?laientBbrosseBbrameBboirasBboasB	battaientBbaroquesBbahBautorit?BaucunsBattirerBasieB
architecteB
aquarellesBapprofonditB
antisth?neB	amoncel?sBamass?BalarmesBaireBaiguBafflueBador?esBabjectB1841B?vanouiB?vad?B?teintsB?panouirB?panouiB?mondeurB?crasaitB?changeronsB?ternelB??aB	?science?B?petitsB	?myst?re?BvoluptueusementBvoirieBvivementBviolenteBviolB
vigoureuseBvignoleBvieillitBvendBveillezBvanneurBvall?esB
valetailleBtympanonBtyBtrivelinBtripotBtremp?BtremblementBtremblantesB
travaill?sB	trasim?neBtransfigureB
tombereauxB	tiendraitBterrass?BtartaneBtapitB	supplicemBsuperbementB	st?rilit?B
souvientonBsouvenezvousBsoutenaientBsouhaitBsonnentBsoixanteBsoeur?B	servitudeBservaitB
semp?trantB
satyressesBsarcophagesBsamuseBsallumerBsaigrissentBsaignaitBsagitaitBr?daitB
r?signetoiBr?pandueBr?busB	rythmiqueBrouvrirBrougirB
roucoulentBros?sBrironsB
retournentBrestaisB
ressemblerB	respironsBrepr?sentantBremisBrelevezvousB	regarderaB	regardentB
refra?chirBrefaitBreculentB
rachitismeB	raccrocheBquestionB	quassist?BprimeursBpressonsBpo?teBposantBportanteBpointueBpointdujourBpoignetBpochesBpipesBpinceB
phyllodoceBpharsaleBper?oitBpens?eBpayantBpatmosBpataugeBpass?esBparaitBpalm?sBouvr?eBota?tiBorph?eB
orf?vrerieBorateurB	ombreusesBoffertsB	obsesseurB	obscurcirBn?ronsBnourrissonsBnombresBnixesBnetteBnentreBnavreBnaus?esBnapol?onBm?dailleBm?chancet?sBmutinBmur?eBmordor?sBmoquesBministreBminaretsBmiletBmignonsBmettronsBmauraBmatouBmassueBmarmotsBmargotonBmapparutB	man?antirBmagiesBmaccableBl?gionBl?ventBl?gliseBlyeuseBlusageBluisB
loublieraiBloquesBlintelligenceBlinstrumentB
lindolenceBlindeBlimprevuBlh?pitalB	lherbetteBlen?treB	lassassinB	lapprocheBlangelusBlabeursB	jimploraiBjaponB
jallumeraiBjailliraBirr?prochableBin?puisableBinv?t?r?BinsultesBinspireB
insidieuseBinou?eBing?nusBingrateBinfolioBinfaillibleBincendieBimpos?eB	imitaientBignorantBh?pitalBhurlaitBhurlaBhommes?BhochetsBharcel?sBhant?sBg?nisseBgu?riteBguindeBguidezBgueusantBglacialBgisantBgentilBgel?sBgalonn?BgallicanBf?brilesB	futaillesBfurtifsBfumerBfuironsBfr?gateBfriperieB	fournaiseBfl?auBfianc?B
feuilleterBferiezBfaunesB	fastueuseBfan?eBfantomeBfangesBfaismoiBfadesB	ex?cuteurBext?nu?eB
extatiquesBexisteBexileBexag?r?eBeussentBetendueB
esp?rancesB	esp?restuB
entretenezBenti?resBentiereBengraisseraB	emprunt?eBembras?sBembrasseB
eauxfortesBd?maisBd?sertesBd?risionBd?lierB	d?gageaitBd?diantB	d?coiff?eBd?chiraientBd?cha?neB	d?calqu?sBd?Bduret?BdonneraiBdonjonBdodoB
docilementBdirentB	dinstantsBdevrionsBdengraisB	davantageBdartreBdaplombB
dallemagneBc?nesBcypr?spourtantBcur?BcupideBcruellesB	croissantBcrimenB	couraientBcorrectBcordagesBconsulteBconnaissantB
condamn?esBcomteBcomptaitBcomblantB	colombierBclavierBcivilisationsBcigogneBcigaleB
chim?riqueBcherchesBcharsBchapitreB
chanteraitBchancelaientBcess?BcausaBcapitoleBb?timentBb?tardsB	bourriqueBbougiesBboueuxB	bocag?resBbl?misBbiblioth?queBbeugleBbelgesBbateauB	balancentBbainsBavouezBavocatBavironBaum?nesBattif?Battabl?sBassur?sB	assouplisBassoupieBarroserBarrang?sBappareillonsBamorisBalt?reBallongeBalbumsBajoutaB	aimezvousB	aiguisaitBaidesB	agonisaitBaffaireB
accueillezBaccrocheBaccompagnaientBaccepteBaccentBabondammentB1833B
?veilletoiB?tablitB
?pouvantesB?piquesB?peronB?lanc?eB?chouerB?chapperB?taientB?clateB?quelqueB?quelB?loi?BxantisB	walpurgisB	vitelliusBvisiterB	vigoureuxBvidantB	versenousBventresBveinerB
vagabondesBtrou?Btrouv?eBtrouvaitB	troublaitBtroublaBtremblementsBtra?naitB	traverserBtransformerB	trahisonsB
tourmente?BtortuesBtitanBtir?BtibulleB	th?ocriteBtardifBtachetteBs?taientilsBs?natBs?coeureBsuspenduBsurtoutjaimeB	sup?rieurBsuairesBstancesBsourdineBsoup?onsBsouhait?B	souffraisBsit?tBsitesB
simulaientBsentonsBsenteurBsenivraB	sarcasmesBsaltimbanqueBsallongeaientBsaliveB
sacrementsBr?vaBr?volteBr?concilierBr?clamerBros?BroiditB
roidissantBrivauxBrid?sBridaitBressouvenirsBrepensBrentraisBrempartB	rejaillirBregarderionsB	regagnantB
redoublaitB	redescendBrecueillaitBrappelBrang?sBrallumerBralentisBrajouterB	raillerieBraboteuxB
quincendieBquharmonieuxBquadrup?desBp?cherBp?merBpyramideBpurifierBpudiquementBpr?par?eB
pr?occuperBpriereBpriaBpress?esBpress?eB	pourpointB	pourceauxBpoudresB	portefauxBpomp?eBpommesB
pleureraisBplaisantB	peupliersBpa?enBpansBpanneBoublieuxBombelleBn?taisB	n?grillonBn?cessairesB
nouveaun?sBnourrisBnerveusementBmyriadesBmusiquesBmugissementB	mouvantesB	mouchardsB
mortuairesB	mortellesBmordusB	montaientBmonstruosit?BminaretBmiltonBmilliersBmilBma?onsB	mangliersBmagiquementBmBluniverselleBlourletBlorsquonBlisteB	lintimit?BlinterminableBlinondeBlimperceptibleBliaientBlexpressionBleurreBlazziBlavoueBlatriumB	lataniersBlargotB	lancettesB
labyrintheBkateBj?taiBj?tudieBj?p?leBjub?B
joignaientBjetteraiBjerseyBjentrevoyaisBirezBinsomnieBinsanit?B	injurieuxBincertainesBimpossiblesBhorriblementBherculeBhaleinesBgu?bresB	gueuserieB
groupaientBgrondentBgrognementsBgravirBgraiss?BgondBgangaBgaminBgagesBfuyardB	fr?quentsBfr?missantsBfr?missanteB	fringanteBfray?sBfrayerBfouaillezmoiBformaitBfl?chirBfleuronsBfleurdelis?B
finalementBfinaleBfestonBfermentsB
fatalementBfalotB	factieuseB	expliciteB	exasp?r?eBexacteBestonB
epieraientBepancheBentrouvraitBenferm?eB	endolorisBencorsonBembaumerBembarrass?sB	d?sempar?Bd?ploy?Bd?pinesBd?moiBd?fenduBd?daleB	d?couvertBd?chiquet?esBdusB	doctrinesBdizainB
distinguerBdisputeBdisentellesB
dinvisibleB	demporterBdavecBdatourBdapollonBdam?resBcr?tinBcro?sBcroisaiBcristallinesB
craindraitBcouv?eBcouverteBcoupezBcorollesBcopi?sB	continuerBconsumeBcombleBcollerB	clitandreBclaquerBclaqueB
chuchotentBchosesl?Bchol?raBchiffonniersBchemiseB	chaufferaBchattesBchang?eB
caressantsB	campanuleB	calentureBb?ninBbroutesB	brouillerBbrill?tBbouhoursBbouffonBbossuBboaBbitumeBbaseB	barbotterBbacioBaveuglementB	autrementBattraperBatilBarchitectesBapparusBanxieuxBalt?r?eB	alouettesBall?sBalerteBalambicBairainBadorerBach?veBaccourirB?ventsB?rablesB?puis?B
?pousset?eB	?missaireB?cumezpartantB	?clairantB?tudiantB?pouvantableB?pilogueB?italieB?grandeB?deB?bahB?amiBzonesByoungBwifeB
voltairienBvivrastuBvitreuxB	visigothsBvirguleB
vindicatifBville?BvilainBve?esBveutil?BvenonsBveniezBvanBvaloisBuniformeB
t?moignageBtyphonsBtr?zenikB	tr?sbraveBtr?fleB
tra?naientBtransper?aitBtranquillementBtrancherBtraduisBtourn?eB
tourment?sB
tortueusesBtonnantBtomberaBtiraitB
tinqui?terBtib?reBth?odoreBtexteBtenfourcherB
tenaillantBtappelleBtallaisBtalentB	taccoudesBs?tesB
s?veillaitB	s?rigeantBs?critBs?mesBs?r?nadeBsylpheBsupprim?B	supposentBsunirB	suivissesBsuivaitBsortaitBsommit?sBsid?ralBshepherdB
sganarelleBsenflammantB	sempresseBsculpteBscrophuleuxBsavatesB
sassombritB
sarrachentBsaronBsaintgermainBsaintespritBsaguetBsaccoudeB
sabaissentBr?v?reB	r?sonnentBr?signationB	r?pandentB
r?conforteB	r?clamaitBr?lerBr?laitBrugissementsBrouvrisBrougesgorgesBrompezB
rocailleuxBriteBrisqueBrichesseBricheletBrichardtroisB
retrouss?sBretombeBrespirationBrespecteBreptileB
rechign?esB	rangeraisBram?neraBramesBramasBraffermirontBrabougriBquindiff?rentBquelquunrendslemoiBquanBp?cheursBp?dantesqueBputoisBpusBpudiqueB	provisionBprosp?reBpromesseB	poussetilB
poussaientBporesBpoorBpontsB	politiqueBplierBplateauBpieds?BpianoBphoquesBpetitsquandBpa?tBpatienteBpasse?BpartitBparticulierB
paraissonsBparachevantBpaillardBoubliezBosseuseB	ossementsBoscilleBoffertesBodesBno?lB	noirecestBnoircirB	nenivrentBneig?BnautileBnappesB
naissaientB	m?lodieuxBm?c?neBmurmurercestBmouvreBmouraitBmoulusBmouleB	moucheronBmorbleuBmont?eBmoisirBmoelleBmobligeBmitreB	mitrailleBmirentB
mettraientBmetteurBmesquinsBmentiBmenezmoiBmatoisB
mapportentBmalinBmacarBl?orBl?opardBluzerneB	londoyantBlogreBlogiveBlisseB
lisolementB	linsens?eBlim?Blimpi?t?BlimpidesBlimbesBlicouBlhostieBleunuqueB	lescaladeBlentendBlencreBlemphaseB
lapportentB	lappelaitB	lanternesBlanguissonsBlaisserastuBlaetitiaBladmirationBlabeilleBj?r?meBjunglesBjoignantBjeunB
jans?nisteBjaillissentBinvit?eBinvent?B	insultantB
insondableB	inqui?tesBimp?rialB	impuret?sB	implorantBh?terBhurlanteBhochantBheurel?Bg?ryonBg?mitBg?missementB
grouillentB	groeculusBgrimpaitBgravieB	gouvernesBgouluBgolgothaBglacialeB	gardefousBgadoueBfuribondBfun?raillesBfuirionsBfr?tilleBfroid?B	frissonnaBfrileuseBform?sB
fleurdelysBfigeB
feuilletonB
feraitcestBfa?onsBfarderB	fantochesBexalteBesp?rantBereint?sB
eperonnantBenvi?B
engourdiesB	enfantinsBemusB	emprunterBecloseBd?sol?sBd?p?eB	d?pouill?Bd?menaitB	d?licatesBd?gouttantsB	d?gloguesB
d?fricheurBd?bord?sBdussestuBdor?BdobscurBdoberonBdivanB
dilettantiBdexhumerB
deviennentB
descendstuBdempruntB
demoiselleBdarriverBdardeursBdardentsB	dantesqueBdamneraientBdadorerB	daccourirBc?tac?Bcur?eBcr?tBcriaientBcraieBcouvrirBcoupleBcort?geBcordonB
convulsiveBcontemplerontBconfusesB
condensaitBconcupiscenceB	comprenezB	comprendsB	composantBcomplaisanteB
comm?ragesBcomblesB	comblatilBcoloraitBclavecinB
clairevoieBcintreBch?riBchoisissantBchildBchassezBceylanB	certitudeB
caverneuseB	cassandreBcarthageB
carmagnoleBcalibanBcachetBbuttantB
buonarottiBbr?l?eBbr?leraientBbrutauxBbruniesBbrumeuseBbrillonsBbourrueBbourgBbonhommeBbluetsB
bleuissantBbastilleBbasilicsBbarineBbarbesBbanvilleBbaleinesBa?riensBavanceB	audessousBatteintB
assur?mentB	assommes?BaspergeBaromeBarideB	appesantiBapparueBappartementBanth?eBanseBannonceB
allonsnousBallianceBail?eBaigreurBagonisaBactricesB	accouplezBaborde
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *#
fR
__inference_<lambda>_38861
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *#
fR
__inference_<lambda>_38866
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?C
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?B
value?BB?B B?B
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	optimizer*
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 

"1
#2
$3*

"0
#1
$2*
* 
?
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 
* 

2serving_default* 
* 
7
3	keras_api
4lookup_table
5token_counts*

6trace_0* 
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
"
embeddings*
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator* 
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator* 
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

#kernel
$bias*

"0
#1
$2*

"0
#1
$2*
* 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
P
\trace_0
]trace_1
^trace_2
_trace_3
`trace_4
atrace_5* 
P
btrace_0
ctrace_1
dtrace_2
etrace_3
ftrace_4
gtrace_5* 
?
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate"m?#m?$m?"v?#v?$v?*
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

rtrace_0* 

strace_0* 
TN
VARIABLE_VALUEembedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

t0
u1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
R
v_initializer
w_create_resource
x_initialize
y_destroy_resource* 
?
z_create_resource
{_initialize
|_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 

"0*

"0*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

#0
$1*

#0
$1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
'
0
1
2
3
4*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
a[
VARIABLE_VALUE	Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE
Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
jd
VARIABLE_VALUEtotal_1Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_1Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
hb
VARIABLE_VALUEtotalIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEcountIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUEAdam/embedding/embeddings/mWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense/kernel/mWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense/bias/mWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/dense/kernel/vWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense/bias/vWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2embedding/embeddingsdense/kernel
dense/bias*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *,
f'R%
#__inference_signature_wrapper_38282
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Adam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_6*%
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *'
f"R 
__inference__traced_save_38969
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasMutableHashTable	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_3count_3total_2count_2total_1count_1totalcountAdam/embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/v*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__traced_restore_39048??
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_37695

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_37853
embedding_input"
embedding_37841:	?N
dense_37847:
dense_37849:
identity??dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_37841*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_37666?
dropout/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37764?
(global_average_pooling1d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37741?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_37847dense_37849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_37695u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????????????
)
_user_specified_nameembedding_input
?%
?
E__inference_sequential_layer_call_and_return_conditional_losses_38682

inputs	3
 embedding_embedding_lookup_38652:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_38652embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/38652*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/38652*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????s
dropout/dropout/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????m
dropout_1/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_37683

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38201
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
sequential_38192:	?N"
sequential_38194:
sequential_38196:
identity??"sequential/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_38192sequential_38194sequential_38196*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37928?
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_37941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?`
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38261
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
sequential_38252:	?N"
sequential_38254:
sequential_38256:
identity??"sequential/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_38252sequential_38254sequential_38256*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38015?
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_37941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_restore_fn_38853
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_37675

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_37928

inputs	3
 embedding_embedding_lookup_37912:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_37912embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/37912*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/37912*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:???????????
dropout/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????x
dropout_1/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?_
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38105

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
sequential_38096:	?N"
sequential_38098:
sequential_38100:
identity??"sequential/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_38096sequential_38098sequential_38100*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38015?
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_37941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
__inference__creator_38816
identity: ??MutableHashTable}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_65*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?%
?
E__inference_sequential_layer_call_and_return_conditional_losses_38015

inputs	3
 embedding_embedding_lookup_37985:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_37985embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/37985*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/37985*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????s
dropout/dropout/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????m
dropout_1/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_388067
3key_value_init1412_lookuptableimportv2_table_handle/
+key_value_init1412_lookuptableimportv2_keys1
-key_value_init1412_lookuptableimportv2_values	
identity??&key_value_init1412/LookupTableImportV2?
&key_value_init1412/LookupTableImportV2LookupTableImportV23key_value_init1412_lookuptableimportv2_table_handle+key_value_init1412_lookuptableimportv2_keys-key_value_init1412_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1412/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init1412/LookupTableImportV2&key_value_init1412/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_37838
embedding_input"
embedding_37826:	?N
dense_37832:
dense_37834:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_37826*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_37666?
dropout/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37675?
(global_average_pooling1d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646?
dropout_1/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37683?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_37832dense_37834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_37695u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????????????
)
_user_specified_nameembedding_input
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_38736

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
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
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????v
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????f
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_38811
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?C
?
__inference_adapt_step_38330
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_38592

inputs3
 embedding_embedding_lookup_38576:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookuph
embedding/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_38576embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/38576*4
_output_shapes"
 :??????????????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/38576*4
_output_shapes"
 :???????????????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :???????????????????
dropout/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*4
_output_shapes"
 :??????????????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????x
dropout_1/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_38550

inputs
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_37941

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_activation_layer_call_and_return_conditional_losses_38692

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????S
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_38714

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37675m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_38783

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_37695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_38762

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_38752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37683`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_37741

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_37823
embedding_input
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????????????
)
_user_specified_nameembedding_input
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_38774

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?_
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_37944

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	#
sequential_37929:	?N"
sequential_37931:
sequential_37933:
identity??"sequential/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_37929sequential_37931sequential_37933*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37928?
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_37941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?~
?
 __inference__wrapped_model_37636
text_vectorization_input\
Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle]
Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	9
5sequential_1_text_vectorization_string_lookup_equal_y<
8sequential_1_text_vectorization_string_lookup_selectv2_t	K
8sequential_1_sequential_embedding_embedding_lookup_37619:	?NN
<sequential_1_sequential_dense_matmul_readvariableop_resource:K
=sequential_1_sequential_dense_biasadd_readvariableop_resource:
identity??4sequential_1/sequential/dense/BiasAdd/ReadVariableOp?3sequential_1/sequential/dense/MatMul/ReadVariableOp?2sequential_1/sequential/embedding/embedding_lookup?Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2y
+sequential_1/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
2sequential_1/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_1/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
4sequential_1/text_vectorization/StaticRegexReplace_1StaticRegexReplace;sequential_1/text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite r
1sequential_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
9sequential_1/text_vectorization/StringSplit/StringSplitV2StringSplitV2=sequential_1/text_vectorization/StaticRegexReplace_1:output:0:sequential_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
?sequential_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
9sequential_1/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_1/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_1/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Asequential_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_1/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_1/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
bsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
osequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
msequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:0Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
3sequential_1/text_vectorization/string_lookup/EqualEqualBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:05sequential_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
6sequential_1/text_vectorization/string_lookup/SelectV2SelectV27sequential_1/text_vectorization/string_lookup/Equal:z:08sequential_1_text_vectorization_string_lookup_selectv2_tTsequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
6sequential_1/text_vectorization/string_lookup/IdentityIdentity?sequential_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????~
<sequential_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4sequential_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
Csequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_1/text_vectorization/RaggedToTensor/Const:output:0?sequential_1/text_vectorization/string_lookup/Identity:output:0Esequential_1/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
sequential_1/sequential/CastCastLsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:???????????
&sequential_1/sequential/embedding/CastCast sequential_1/sequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
2sequential_1/sequential/embedding/embedding_lookupResourceGather8sequential_1_sequential_embedding_embedding_lookup_37619*sequential_1/sequential/embedding/Cast:y:0*
Tindices0*K
_classA
?=loc:@sequential_1/sequential/embedding/embedding_lookup/37619*,
_output_shapes
:??????????*
dtype0?
;sequential_1/sequential/embedding/embedding_lookup/IdentityIdentity;sequential_1/sequential/embedding/embedding_lookup:output:0*
T0*K
_classA
?=loc:@sequential_1/sequential/embedding/embedding_lookup/37619*,
_output_shapes
:???????????
=sequential_1/sequential/embedding/embedding_lookup/Identity_1IdentityDsequential_1/sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:???????????
(sequential_1/sequential/dropout/IdentityIdentityFsequential_1/sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:???????????
Gsequential_1/sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
5sequential_1/sequential/global_average_pooling1d/MeanMean1sequential_1/sequential/dropout/Identity:output:0Psequential_1/sequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
*sequential_1/sequential/dropout_1/IdentityIdentity>sequential_1/sequential/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:??????????
3sequential_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp<sequential_1_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
$sequential_1/sequential/dense/MatMulMatMul3sequential_1/sequential/dropout_1/Identity:output:0;sequential_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
4sequential_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%sequential_1/sequential/dense/BiasAddBiasAdd.sequential_1/sequential/dense/MatMul:product:0<sequential_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1/activation/SigmoidSigmoid.sequential_1/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#sequential_1/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp5^sequential_1/sequential/dense/BiasAdd/ReadVariableOp4^sequential_1/sequential/dense/MatMul/ReadVariableOp3^sequential_1/sequential/embedding/embedding_lookupL^sequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2l
4sequential_1/sequential/dense/BiasAdd/ReadVariableOp4sequential_1/sequential/dense/BiasAdd/ReadVariableOp2j
3sequential_1/sequential/dense/MatMul/ReadVariableOp3sequential_1/sequential/dense/MatMul/ReadVariableOp2h
2sequential_1/sequential/embedding/embedding_lookup2sequential_1/sequential/embedding/embedding_lookup2?
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_sequential_1_layer_call_fn_37961
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_37944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
T
8__inference_global_average_pooling1d_layer_call_fn_38741

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_38561

inputs	
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37928o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_38572

inputs	
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_38709

inputs)
embedding_lookup_38703:	?N
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_38703Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/38703*4
_output_shapes"
 :??????????????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/38703*4
_output_shapes"
 :???????????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :???????????????????
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_38757

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37741o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_37702

inputs"
embedding_37667:	?N
dense_37696:
dense_37698:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_37667*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_37666?
dropout/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37675?
(global_average_pooling1d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646?
dropout_1/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37683?
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_37696dense_37698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_37695u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_38793

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
*
__inference_<lambda>_38866
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
,__inference_sequential_1_layer_call_fn_38355

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_37944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_37666

inputs)
embedding_lookup_37660:	?N
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_37660Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/37660*4
_output_shapes"
 :??????????????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/37660*4
_output_shapes"
 :???????????????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :???????????????????
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
~
)__inference_embedding_layer_call_fn_38699

inputs
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_37666|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
:
__inference__creator_38798
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1413*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
.
__inference__initializer_38821
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?_
?
!__inference__traced_restore_39048
file_prefix8
%assignvariableop_embedding_embeddings:	?N1
assignvariableop_1_dense_kernel:+
assignvariableop_2_dense_bias:M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: &
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: $
assignvariableop_8_total_3: $
assignvariableop_9_count_3: %
assignvariableop_10_total_2: %
assignvariableop_11_count_2: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: #
assignvariableop_14_total: #
assignvariableop_15_count: B
/assignvariableop_16_adam_embedding_embeddings_m:	?N9
'assignvariableop_17_adam_dense_kernel_m:3
%assignvariableop_18_adam_dense_bias_m:B
/assignvariableop_19_adam_embedding_embeddings_v:	?N9
'assignvariableop_20_adam_dense_kernel_v:3
%assignvariableop_21_adam_dense_bias_v:
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2		[
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
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:3RestoreV2:tensors:4*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 ]

Identity_3IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_8IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_3Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
,
__inference__destroyer_38826
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
`
'__inference_dropout_layer_call_fn_38719

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37764|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_38282
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *)
f$R"
 __inference__wrapped_model_37636o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?o
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38444

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	>
+sequential_embedding_embedding_lookup_38427:	?NA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
sequential/CastCast?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????x
sequential/embedding/CastCastsequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_38427sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/38427*,
_output_shapes
:??????????*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/38427*,
_output_shapes
:???????????
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:???????????
sequential/dropout/IdentityIdentity9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????|
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/global_average_pooling1d/MeanMean$sequential/dropout/Identity:output:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
sequential/dropout_1/IdentityIdentity1sequential/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:??????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense/MatMulMatMul&sequential/dropout_1/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
activation/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38528

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	>
+sequential_embedding_embedding_lookup_38497:	?NA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*
pattern<br />*
rewrite ?
'text_vectorization/StaticRegexReplace_1StaticRegexReplace.text_vectorization/StaticRegexReplace:output:0*#
_output_shapes
:?????????*A
pattern64[!"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV20text_vectorization/StaticRegexReplace_1:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
sequential/CastCast?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????x
sequential/embedding/CastCastsequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_38497sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/38497*,
_output_shapes
:??????????*
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/38497*,
_output_shapes
:???????????
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????e
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sequential/dropout/dropout/MulMul9sequential/embedding/embedding_lookup/Identity_1:output:0)sequential/dropout/dropout/Const:output:0*
T0*,
_output_shapes
:???????????
 sequential/dropout/dropout/ShapeShape9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0n
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:???????????
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:???????????
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????|
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/global_average_pooling1d/MeanMean$sequential/dropout/dropout/Mul_1:z:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????g
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 sequential/dropout_1/dropout/MulMul1sequential/global_average_pooling1d/Mean:output:0+sequential/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:??????????
"sequential/dropout_1/dropout/ShapeShape1sequential/global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:?
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/dense/MatMulMatMul&sequential/dropout_1/dropout/Mul_1:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
activation/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????e
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_38647

inputs	3
 embedding_embedding_lookup_38631:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_38631embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/38631*,
_output_shapes
:??????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/38631*,
_output_shapes
:???????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:???????????
dropout/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????x
dropout_1/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_activation_layer_call_fn_38687

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_37941`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_1_layer_call_fn_38374

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_38105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_388617
3key_value_init1412_lookuptableimportv2_table_handle/
+key_value_init1412_lookuptableimportv2_keys1
-key_value_init1412_lookuptableimportv2_values	
identity??&key_value_init1412/LookupTableImportV2?
&key_value_init1412/LookupTableImportV2LookupTableImportV23key_value_init1412_lookuptableimportv2_table_handle+key_value_init1412_lookuptableimportv2_keys-key_value_init1412_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1412/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init1412/LookupTableImportV2&key_value_init1412/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_37764

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
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
 :??????????????????|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????v
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????f
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
__inference_save_fn_38845
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
*__inference_sequential_layer_call_fn_37711
embedding_input
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????????????
)
_user_specified_nameembedding_input
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_38724

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :??????????????????h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :??????????????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_37803

inputs"
embedding_37791:	?N
dense_37797:
dense_37799:
identity??dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_37791*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_37666?
dropout/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_37764?
(global_average_pooling1d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_37646?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_37741?
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_37797dense_37799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_37695u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_38539

inputs
unknown:	?N
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_37702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_38747

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_1_layer_call_fn_38141
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_38105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?%
?
E__inference_sequential_layer_call_and_return_conditional_losses_38626

inputs3
 embedding_embedding_lookup_38596:	?N6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?embedding/embedding_lookuph
embedding/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_38596embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/38596*4
_output_shapes"
 :??????????????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/38596*4
_output_shapes"
 :???????????????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????s
dropout/dropout/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :???????????????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :???????????????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMeandropout/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????m
dropout_1/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense/MatMulMatMuldropout_1/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?7
?	
__inference__traced_save_38969
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_6

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
:*
dtype0*?
value?B?B&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B ?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *'
dtypes
2		?
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
_input_shapesy
w: :	?N::::: : : : : : : : : : : : : :	?N:::	?N::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0?????????@

activation2
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	optimizer"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
5
"1
#2
$3"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
?
*trace_0
+trace_1
,trace_2
-trace_32?
,__inference_sequential_1_layer_call_fn_37961
,__inference_sequential_1_layer_call_fn_38355
,__inference_sequential_1_layer_call_fn_38374
,__inference_sequential_1_layer_call_fn_38141?
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
 z*trace_0z+trace_1z,trace_2z-trace_3
?
.trace_0
/trace_1
0trace_2
1trace_32?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38444
G__inference_sequential_1_layer_call_and_return_conditional_losses_38528
G__inference_sequential_1_layer_call_and_return_conditional_losses_38201
G__inference_sequential_1_layer_call_and_return_conditional_losses_38261?
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
 z.trace_0z/trace_1z0trace_2z1trace_3
?B?
 __inference__wrapped_model_37636text_vectorization_input"?
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
"
	optimizer
,
2serving_default"
signature_map
"
_generic_user_object
L
3	keras_api
4lookup_table
5token_counts"
_tf_keras_layer
?
6trace_02?
__inference_adapt_step_38330?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z6trace_0
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
"
embeddings"
_tf_keras_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C_random_generator"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator"
_tf_keras_layer
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
\trace_0
]trace_1
^trace_2
_trace_3
`trace_4
atrace_52?
*__inference_sequential_layer_call_fn_37711
*__inference_sequential_layer_call_fn_38539
*__inference_sequential_layer_call_fn_38550
*__inference_sequential_layer_call_fn_37823
*__inference_sequential_layer_call_fn_38561
*__inference_sequential_layer_call_fn_38572?
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
 z\trace_0z]trace_1z^trace_2z_trace_3z`trace_4zatrace_5
?
btrace_0
ctrace_1
dtrace_2
etrace_3
ftrace_4
gtrace_52?
E__inference_sequential_layer_call_and_return_conditional_losses_38592
E__inference_sequential_layer_call_and_return_conditional_losses_38626
E__inference_sequential_layer_call_and_return_conditional_losses_37838
E__inference_sequential_layer_call_and_return_conditional_losses_37853
E__inference_sequential_layer_call_and_return_conditional_losses_38647
E__inference_sequential_layer_call_and_return_conditional_losses_38682?
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
 zbtrace_0zctrace_1zdtrace_2zetrace_3zftrace_4zgtrace_5
?
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate"m?#m?$m?"v?#v?$v?"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?
rtrace_02?
*__inference_activation_layer_call_fn_38687?
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
 zrtrace_0
?
strace_02?
E__inference_activation_layer_call_and_return_conditional_losses_38692?
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
 zstrace_0
':%	?N2embedding/embeddings
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_1_layer_call_fn_37961text_vectorization_input"?
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
,__inference_sequential_1_layer_call_fn_38355inputs"?
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
,__inference_sequential_1_layer_call_fn_38374inputs"?
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
?B?
,__inference_sequential_1_layer_call_fn_38141text_vectorization_input"?
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
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38444inputs"?
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
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38528inputs"?
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
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38201text_vectorization_input"?
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
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38261text_vectorization_input"?
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
#__inference_signature_wrapper_38282text_vectorization_input"?
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
"
_generic_user_object
f
v_initializer
w_create_resource
x_initialize
y_destroy_resourceR jtf.StaticHashTable
L
z_create_resource
{_initialize
|_destroy_resourceR Z

 ??
?B?
__inference_adapt_step_38330iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_embedding_layer_call_fn_38699?
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
 z?trace_0
?
?trace_02?
D__inference_embedding_layer_call_and_return_conditional_losses_38709?
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
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
'__inference_dropout_layer_call_fn_38714
'__inference_dropout_layer_call_fn_38719?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_38724
B__inference_dropout_layer_call_and_return_conditional_losses_38736?
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
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
8__inference_global_average_pooling1d_layer_call_fn_38741?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_38747?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_1_layer_call_fn_38752
)__inference_dropout_1_layer_call_fn_38757?
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
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38762
D__inference_dropout_1_layer_call_and_return_conditional_losses_38774?
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
 z?trace_0z?trace_1
"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_dense_layer_call_fn_38783?
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
 z?trace_0
?
?trace_02?
@__inference_dense_layer_call_and_return_conditional_losses_38793?
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
 z?trace_0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_sequential_layer_call_fn_37711embedding_input"?
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
*__inference_sequential_layer_call_fn_38539inputs"?
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
*__inference_sequential_layer_call_fn_38550inputs"?
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
?B?
*__inference_sequential_layer_call_fn_37823embedding_input"?
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
*__inference_sequential_layer_call_fn_38561inputs"?
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
*__inference_sequential_layer_call_fn_38572inputs"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_38592inputs"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_38626inputs"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_37838embedding_input"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_37853embedding_input"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_38647inputs"?
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
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_38682inputs"?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
?B?
*__inference_activation_layer_call_fn_38687inputs"?
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
E__inference_activation_layer_call_and_return_conditional_losses_38692inputs"?
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
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_38798?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_38806?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_38811?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_38816?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_38821?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_38826?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
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
?B?
)__inference_embedding_layer_call_fn_38699inputs"?
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
D__inference_embedding_layer_call_and_return_conditional_losses_38709inputs"?
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
?B?
'__inference_dropout_layer_call_fn_38714inputs"?
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
?B?
'__inference_dropout_layer_call_fn_38719inputs"?
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
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_38724inputs"?
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
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_38736inputs"?
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
?B?
8__inference_global_average_pooling1d_layer_call_fn_38741inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_38747inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_1_layer_call_fn_38752inputs"?
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
?B?
)__inference_dropout_1_layer_call_fn_38757inputs"?
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
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38762inputs"?
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
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38774inputs"?
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
?B?
%__inference_dense_layer_call_fn_38783inputs"?
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
@__inference_dense_layer_call_and_return_conditional_losses_38793inputs"?
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
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_38798"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_38806"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_38811"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_38816"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_38821"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_38826"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
,:*	?N2Adam/embedding/embeddings/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*	?N2Adam/embedding/embeddings/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
?B?
__inference_save_fn_38845checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_38853restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant6
__inference__creator_38798?

? 
? "? 6
__inference__creator_38816?

? 
? "? 8
__inference__destroyer_38811?

? 
? "? 8
__inference__destroyer_38826?

? 
? "? A
__inference__initializer_388064???

? 
? "? :
__inference__initializer_38821?

? 
? "? ?
 __inference__wrapped_model_37636?
4???"#$=?:
3?0
.?+
text_vectorization_input?????????
? "7?4
2

activation$?!

activation??????????
E__inference_activation_layer_call_and_return_conditional_losses_38692X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_activation_layer_call_fn_38687K/?,
%?"
 ?
inputs?????????
? "??????????j
__inference_adapt_step_38330J5???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
@__inference_dense_layer_call_and_return_conditional_losses_38793\#$/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_38783O#$/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_38762\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_38774\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_1_layer_call_fn_38752O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_1_layer_call_fn_38757O3?0
)?&
 ?
inputs?????????
p
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_38724v@?=
6?3
-?*
inputs??????????????????
p 
? "2?/
(?%
0??????????????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_38736v@?=
6?3
-?*
inputs??????????????????
p
? "2?/
(?%
0??????????????????
? ?
'__inference_dropout_layer_call_fn_38714i@?=
6?3
-?*
inputs??????????????????
p 
? "%?"???????????????????
'__inference_dropout_layer_call_fn_38719i@?=
6?3
-?*
inputs??????????????????
p
? "%?"???????????????????
D__inference_embedding_layer_call_and_return_conditional_losses_38709q"8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
)__inference_embedding_layer_call_fn_38699d"8?5
.?+
)?&
inputs??????????????????
? "%?"???????????????????
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_38747{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_38741nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????y
__inference_restore_fn_38853Y5K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_38845?5&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38201z
4???"#$E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38261z
4???"#$E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38444h
4???"#$3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_38528h
4???"#$3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_37961m
4???"#$E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_38141m
4???"#$E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_38355[
4???"#$3?0
)?&
?
inputs?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_38374[
4???"#$3?0
)?&
?
inputs?????????
p

 
? "???????????
E__inference_sequential_layer_call_and_return_conditional_losses_37838w"#$I?F
??<
2?/
embedding_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_37853w"#$I?F
??<
2?/
embedding_input??????????????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_38592n"#$@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_38626n"#$@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_38647f"#$8?5
.?+
!?
inputs??????????	
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_38682f"#$8?5
.?+
!?
inputs??????????	
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_37711j"#$I?F
??<
2?/
embedding_input??????????????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_37823j"#$I?F
??<
2?/
embedding_input??????????????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_38539a"#$@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_38550a"#$@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_38561Y"#$8?5
.?+
!?
inputs??????????	
p 

 
? "???????????
*__inference_sequential_layer_call_fn_38572Y"#$8?5
.?+
!?
inputs??????????	
p

 
? "???????????
#__inference_signature_wrapper_38282?
4???"#$Y?V
? 
O?L
J
text_vectorization_input.?+
text_vectorization_input?????????"7?4
2

activation$?!

activation?????????