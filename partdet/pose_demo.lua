--require 'pl'
--require 'qt'
--require 'qtwidget'
--require 'qtuiloader'
require 'image'
--require 'nnx'
require 'torch'
--require 'cutorch'
--require 'nn'
require 'cunn'
-- require('net-toolkit')
-- require('mobdebug').start()

opt = {}
opt.network = 'results_all/model_22.net'
opt.threads = 1
opt.batchSize = 2
opt.type = 'cuda'

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(2)

local function split(str, sep)
  sep = sep or ','
  fields={}
  local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
  if not matchfunc then return {str} end
  for str in matchfunc do
      table.insert(fields, str)
  end
  return fields
end

local function readCSV(filename,ht,wid)
  local csv_ten = torch.Tensor(ht,wid)
  local file = io.open(filename,"r")
  local ln=1
  for line in file:lines() do 
    temp = split(line,",")
    for i=1,#temp do
      csv_ten[ln][i] = tonumber(temp[i])
    end
    ln=ln+1
  end
  file:close()
  return csv_ten
end

--InCoords - (x,y), index begining with 1
local function mapInCoord2OutCoord(x,y)
 --zeropad
  x=x+50; 
  y=y+30;
  --conv+relu+pool
  x=x-(5-1)/2; x=(x-1)/2+1
  y=y-(5-1)/2; y=(y-1)/2+1
  --conv+relu+pool
  x=x-(5-1)/2; x=(x-1)/2+1
  y=y-(5-1)/2; y=(y-1)/2+1
  --conv+relu
  x=x-(5-1)/2;
  y=y-(5-1)/2;
  --conv+relu
  x=x-(9-1)/2;
  y=y-(9-1)/2;
  
  return {x,y}
end

function makeGroundTruth(coords)
  outmap = torch.Tensor(4,60,90)
  local params = {}
  params.width=90; params.height=60; params.amplitude=30
  params.sigma_horz=1/90; params.sigma_vert=1/60;
  for i=1,8,2 do
    cent = mapInCoord2OutCoord(coords[i],coords[i+1])
    params.mean_horz=(cent[1]-1)/89; params.mean_vert=(cent[2]-1)/59;
    outmap[torch.ceil(i/2)] = image.gaussian(params)
  end
  return outmap
end

-- load pre-trained network from disk
-- model = netToolkit.loadNet(opt.network) --load a network split in two: network and classifier
model = torch.load(opt.network)

local startImages = 4000
local endImages = 7974
local numImages = 7974
local img = torch.Tensor(3,240,320)
local inputs = torch.Tensor(opt.batchSize,3,240,320) -- get size from data
if opt.type == 'cuda' then 
   inputs = inputs:cuda()
end

torch.setnumthreads(1)

print("ExCoords reading..")
local ExCoords = readCSV('../data/FLIC_dataset/FLIC/train/excoords.csv',numImages,8)
print("ExCoords read!")

local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian(9)):float()
local channels = {'r','g','b'}
for f=startImages,endImages do
  img = image.load('../data/FLIC_dataset/FLIC/train/images/'..f..'.jpg') 
  for c in ipairs(channels) do
    img[{ {c},{},{} }] = normalization:forward(img[{ {c},{},{} }])
  end
  
  for i = 1,opt.batchSize do
    inputs[i] = img
  end
  
  local preds = model:forward(inputs)

  local gt = makeGroundTruth(ExCoords[f])
  print(img:size())
  print(image.scale(gt[{ {2,4} }],320,240):size())
  image.display( { image={img,image.scale(gt[{ {2,4} }],320,240),image.scale(preds[{1, {2,4} }]:float(),320,240)} , padding=40 , scaleeach=true } )
  io.read(1)
end

