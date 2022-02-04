# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:10:45 2021

@author: RAPY
"""

#%% IMPORTS
from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray #used when calling ncdata.rio.write_crs
import os
import matplotlib.pyplot as plt
import netCDF4 #not directly used but needs to be imported for some nc4 files manipulations
import numpy as np
import io
import unicodedata
idx=pd.IndexSlice #pandas index slicer
#%% INPUTS
#Files - general working directory
folder_path=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS'

#Basin file corresponds to the limits of the River Basin 
#(you might already wnat to account for the fact that they might be several river basins in the study area)
basin_file=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\official\Area_estudio_Amz_fixed.shp'

#Catchment file, corresponds to the hydrological and agricultural resolution in the model
#There is a world-wide catchment (and river basin) data - see  https://www.hydrosheds.org/
catch_file=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\amazon_catchment_HBID_v2_wcountry_cleaned.shp'
#catch_file=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\amazon_countries_clipped.shp'
#catch_file=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\Countries_WGS84.shp'

#Projection crs - IT is VERY important to ensure that all data have the same projection when performing calculations
bCRS='EPSG:4326'

#plot options
FULLWIDTH=6.69 #inches (21 - 4 cm margin = 17 cm)
FULLLEN=9.33 #inches (29.7 - 6 cm margin = 23.7 cm)
DPI=300

#%% Functions
#CLIP NC file: open nc file and clip by basin_shape
def clip_raster(input_nc,basin_file,clip_file,bCRS='EPSG:4326',variable=None,t0=None):
    basin_shape = gpd.read_file(basin_file).to_crs(bCRS)
    with rioxarray.open_rasterio(input_nc,decode_times=False,variable=variable) as ncdata:
        ncdata.rio.write_crs(bCRS, inplace=True)
        clipped = ncdata.rio.clip(basin_shape.geometry, basin_shape.crs, drop=True) #.geometry.apply(mapping)
        clipped.to_netcdf(clip_file)#,format='NETCDF4',engine='netcdf4')
    return

#get first time step value of nc file
def get_t0(input_nc,variable):
    with rioxarray.open_rasterio(input_nc,decode_times=False,variable=variable) as ncdata:
        t0=ncdata.time.values[0]
    return t0

#export to excel - In general csv to be prefered over excel
def export_to_excel(outpath,data):   
    writer = pd.ExcelWriter(outpath, engine='openpyxl') #open excel file to dump results
    if 'geometry' in data.columns:
        data=data.drop(axis=1,labels='geometry')
    data.to_excel(writer) #export geopanda dataframe to excel file
    writer.save() #save file
    writer.close()

#Remove accents from strings - to avoid troubles in python
def remove_accent(string):
    return unicodedata.normalize('NFKD', string).encode('ASCII', 'ignore').decode('utf8')

# Convert from raster.nc to area (catchments)
#Uses .nc(4) file as input and outputs and calculate stats for catch_file
#stats can be mean, sum, max, ... all touched=True can only be used for mean
#eventually clips it around a specific zone (warning - might not improve efficiency of calculations)
#different time steps are either added as columns, or the output has time as index with addt2idx=True
def netcdf_to_shape(input_nc,catch_file,clip_file=None,basin_file=None,bCRS='EPSG:4326',
                    variable=None,all_touched=False,stats='sum',CF=1, 
                    t0=None,refyear=0,sel_time=None,cl_name='stats_',addt2idx=False):

    if clip_file != None: #Clip the file to basin boundaries
        clip_raster(input_nc,basin_file,clip_file,bCRS=bCRS,variable=variable)
        input_nc=clip_file #Set file to use as original nc file

    if t0==None: #et t0 in nc file
        t0=get_t0(input_nc,variable)
            
    #shapefile to store data in
    catch_shape = gpd.read_file(catch_file).to_crs(bCRS)
    
    #get nc file bands (usually time steps)
    with rasterio.open(input_nc) as src:
         band_indexes = src.indexes
         
    #create dataframe recieving results
    if addt2idx == False:
        output=catch_shape
    else:
        #create output dataframe
        tidx = sel_time if sel_time!=None else band_indexes
        iterable=[catch_shape[addt2idx].values,tidx]
        mindex=pd.MultiIndex.from_product(iterable, names=[ 'id','time'])
        output=pd.DataFrame(index=mindex,columns=[cl_name])
        idx=pd.IndexSlice
        
    for b in band_indexes: #get all bands
        if refyear!=0: #transform time index
            time=int(refyear+t0+b-1)
        else:
            time=b
        if sel_time == None or time in sel_time:
            band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, 
                                                raster=input_nc,
                                                band=b,
                                                all_touched=all_touched,
                                                stats=stats))
            if addt2idx==False:
                output[cl_name+str(time)] = band_stats[stats]*CF
            else:
                output.loc[idx[:,time],cl_name] = band_stats[stats].values*CF
    
    return output

#%% Calculations
#shapefile to store data in
idname='SUB_BAS' #what is the id field of the shapefile
catch_shape = gpd.read_file(catch_file).to_crs(bCRS)
catch_shape['ncatch']=catch_shape[idname].apply(lambda k: 'c'+str(k))
#catch_shape['catch_ds']=catch_shape['TO_BAS'].apply(lambda k: 'c'+str(k))
#catch_shape=catch_shape.rename({'CNTRY_NAME':'ncountry'},axis=1)
#catch_shape=catch_shape.drop(['fid','cat','LEGEND','__median'],axis=1)
#catch_shape = catch_shape[['SUB_BAS','geometry']]
#catch_shape.to_file(r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\Countries_WGS84.shp')
area_km2 = catch_shape.to_crs({'proj':'cea'}).area/10**6 #area in km2  
catch_shape['area_km2'] = area_km2

#%% Livestock (cattle) 
#source: https://dataverse.harvard.edu/dataverse/glw or https://www.fao.org/livestock-systems/global-distributions/en/
#data: 
raster=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Agriculture\Livestock\5_Ct_2010_Da.tif'
band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, raster=raster, 
                                    all_touched=False, stats='sum'))
catch_shape['cattle_upkm2'] = band_stats['sum']/area_km2



#%% Population
#source: https://doi.org/10.7927/q7z9-9r69
#download: https://sedac.ciesin.columbia.edu/data/set/popdynamics-1-km-downscaled-pop-base-year-projection-ssp-2000-2100-rev01
folder=r'C:\Users\RAPY\OneDrive - COWI\Data\GlobalPopulationProjections'
for scen in ['SSP1','SSP3','SSP5']:
    for year in [2020,2030,2040,2050]:
        print(year)
        data=os.path.join(scen,r'Total\ASCII',scen+'_'+str(year)+'.txt')
        raster=os.path.join(folder,data)
        band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, raster=raster, 
                                            all_touched=False, stats='sum'))
        catch_shape[scen+'_'+str(year)]=band_stats
catch_shape=catch_shape.sort_index(axis=1)

#%% Population - DO NOT USE THIS ONE
input_nc=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\ISMIP\population\population_rcp26soc_0p5deg_annual_2006-2099.nc4'
sel_time=[2020,2030,2050]
pop=netcdf_to_shape(input_nc,catch_file,clip_file=None,basin_file=None,bCRS='EPSG:4326',
                    variable=None,all_touched=False,stats='sum',CF=1, 
                    t0=146,refyear=1860,sel_time=sel_time,cl_name='pop_')
catch_shape=pd.concat([catch_shape,pop[['pop_'+str(y) for y in sel_time]]],
                      axis=1)
#%% Power plants
#source: http://datasets.wri.org/dataset/globalpowerplantdatabase
from shapely.geometry import Point
path = r'C:\Users\RAPY\OneDrive - COWI\Data\global_power_plant_database_v_1_3\global_power_plant_database.csv'
pdata = pd.read_csv(path)
countries = ['BOL','BRA','PER','COL','ECU','SUR','GUY','VEN']
pdata = pdata[pdata['country'].isin(countries)]
geometry = [Point(xy) for xy in zip(pdata.longitude, pdata.latitude)]
gpdata = gpd.GeoDataFrame(pdata, crs=bCRS, geometry=geometry)
gpdata_amz=gpd.clip(gpdata,gpd.read_file(basin_file).to_crs(bCRS))
#agregate
gpdata_ag=pd.DataFrame(gpdata.groupby(['country','primary_fuel']).sum()) 
gpdata_amz_ag=pd.DataFrame(gpdata_amz.groupby(['country','primary_fuel']).sum())
gpdata_ag['commissioning_year']=gpdata.groupby(['country','primary_fuel']).mean()['commissioning_year']
gpdata_amz_ag['commissioning_year']=gpdata_amz.groupby(['country','primary_fuel']).mean()['commissioning_year']
#difference
d=gpdata_ag-gpdata_amz_ag
d[d.isnull()]=gpdata_ag[d.isnull()]
gpdata_noamz_ag=d
#Export
gpdata_ag[['capacity_mw','commissioning_year']].to_csv(os.path.join(folder_path,'powercap_country.csv'))
gpdata_amz_ag[['capacity_mw','commissioning_year']].to_csv(os.path.join(folder_path,'powercap_country_amazon.csv'))
gpdata_noamz_ag[['capacity_mw','commissioning_year']].to_csv(os.path.join(folder_path,'powercap_country_noamazon.csv'))
#catch_shape = gpd.read_file(catch_file).to_crs(bCRS)

#%% Tree cover
raster=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Forest_loss_year\amazon_tree_cover_epoch2015_v202.tif'
band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, raster=raster, 
                                    all_touched=True, stats='mean'))
catch_shape['treecover_p']=band_stats

#%% deforestation - Did not get it to work for now
def l1_10(x):
    x=np.array(x)
    return ((0 < x) & (x <= 10)).sum()
    #return sum([1 for k in x if 0<=k<=10])
def l11_15(x):
    return sum([1 for k in x if 10<k<=15])
def l16_20(x):
    return sum([1 for k in x if 15<k<=20])
raster=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Forest_loss_year\Hansen_GFC_2020_lossyear_merged.tif'
band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, raster=raster,
                                    all_touched=False, #stats='sum'))#,
                                    add_stats={'loss1_10':l1_10}))#,
                                               #'loss11_15':l11_15,'loss16_20':l16_20}))
                                               
#%% deforestation TEST
#source: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/download.html
# File path
dem_fp = r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Forest_loss_year\Hansen_GFC_2020_lossyear_merged.tif'
# Read the Digital Elevation Model for Helsinki
dem = rasterio.open(dem_fp, dtype='uint8')
# Read the raster values
array = dem.read(1)
# Get the affine
affine = dem.transform
# Calculate zonal statistics for Kallio
zs_kallio = zonal_stats(catch_shape, array, affine=affine, stats=['min'], dtype='uint8')           

#%% Soybeans
#this was just for mapping example, see : https://github.com/RaphaelPB/SPAM_data_reader (maybe keep for later stage)
raster=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Agriculture\SPAM_Soybean_rainfed_harvested_clipped.tif'
band_stats=pd.DataFrame(zonal_stats(vectors=catch_shape, raster=raster, 
                                    all_touched=False, stats='sum'))/1000
catch_shape['soy_kha']=band_stats


#%% YIELD
#source: https://data.isimip.org/search/tree/ISIMIP2b%2FOutputData%2Fagriculture/variable/yield/
#note: there are various crop yield models, climate and socio-economic scenarios
#technically you do not need to download that data, you can read directly from server

sel_time=range(2015,2056)
idxname='CNTRY_NAME' #'SUB_BAS' #
raster=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\ISMIP\Yield\clm45_gfdl-esm2m_ewembi_rcp26_2005soc_co2_yield-soy-noirr_global_annual_2006_2099.nc4'
models=['clm45','gepic','lpjml','pepic']
miidx=pd.MultiIndex.from_product([catch_shape['CNTRY_NAME'].values,sel_time],names=['country','year'])
yldp=pd.DataFrame(index=miidx,columns=models)
  
for m in models:
    rstr=raster.split('clm45')[0]+m+raster.split('clm45')[1] #this changes the model
    if os.path.exists(rstr):
        yld=netcdf_to_shape(rstr,catch_file,clip_file=None,basin_file=None,bCRS='EPSG:4326',
                            variable=None,all_touched=True,stats='mean',CF=1, 
                            t0=345,refyear=1661,sel_time=sel_time,cl_name='')    
        for y in sel_time:
            yldp.loc[idx[:,y],m]=yld[str(y)].values

#average/max
years=[2020,2030,2040,2050]
miidx=pd.MultiIndex.from_product([catch_shape[idxname].values,years],names=['country','year'])        
maxyld=pd.DataFrame(index=miidx,columns=models)
avyld=pd.DataFrame(index=miidx,columns=models)
for y in years:
    yrs=[ky for ky in range(y-5,y+5)]
    maxyld.loc[idx[:,y],:]=yldp.loc[idx[:,yrs],:].groupby('country').max().values
    avyld.loc[idx[:,y],:]=yldp.loc[idx[:,yrs],:].groupby('country').mean().values

#%%Runoff, Rainfall, ET
#source: https://data.isimip.org/search/tree/ISIMIP2b%2FOutputData%2Fwater_regional/
#similar to yield, multiple models and scenarios

ISRUNOFF=False
cl_name='rainfallMm3pkm2' #'rainfall_mmpmonth'#
idxname='CNTRY_NAME' #'SUB_BAS' #
#input_nc=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\ISMIP\Runoff\clm45_gfdl-esm2m_ewembi_rcp26_2005soc_co2_qtot_global_monthly_2006_2099.nc4'
input_nc=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\ISMIP\Runoff\clm45_gfdl-esm2m_ewembi_rcp26_2005soc_co2_rainf_global_monthly_2006_2099.nc4'
sel_time=range(0,12)
t0=4140 #first time step in data (jan 2006)
refyear=-4308 #So that 1 jan 2020 is t=0

if ISRUNOFF:
    CF= 10**-9*24*3600*365/12*10**6 #kg.s-1.m-2 to Mm3/km2/month
else:
    CF= 10**-9*24*3600*365/12*10**6 *1000 #kg.s-1.m-2 to mm/month
param=netcdf_to_shape(input_nc,catch_file,clip_file=None,basin_file=None,bCRS='EPSG:4326',
                       variable=None,all_touched=True,stats='mean',CF=CF, 
                       t0=t0,refyear=refyear,sel_time=sel_time,cl_name=cl_name,addt2idx=idxname)
#Get runoff in Mm3/month (from Mm3/km2/month)
param_net=param
if ISRUNOFF:
    area=catch_shape.set_index(idxname)['area_km2']
    for t in sel_time:
        param_net.loc[idx[:,t],cl_name]=param.loc[idx[:,t],cl_name]*area.values
#Transform to what-if format
param_wi=pd.DataFrame(index=sel_time, columns=catch_shape[idxname].values)
for idd in catch_shape[idxname].values:
    param_wi.loc[:,idd]=param_net.loc[idx[idd,:],cl_name].values

#%%Collect reservoirs
#source: GRAND database, global database - http://globaldamwatch.org/grand/ (download link does not work lately, but available on DTU server)
#source for south america: https://essd.copernicus.org/articles/13/213/2021/essd-13-213-2021-assets.html
    
MINCAP=50 #minimum storing capacity to consider (in Mm3-depends on data)
#Import reservoirs
path = r'C:\Users\RAPY\OneDrive - COWI\Data\6_ddsa_dams\DDSA.shp'
reservoirs=gpd.read_file(path)
reservoirs.crs=bCRS
reservoirs=gpd.clip(reservoirs,catch_shape)
reservoirs.rename(columns={'Name_of_th':'nres','Reservoir':'wStorCap'},inplace=True)
#filter reservoirs under threshold
reservoirs=reservoirs[reservoirs['wStorCap']>MINCAP]
#remove accents
reservoirs['nres']=reservoirs['nres'].map(remove_accent)
#get reservoir catchment
reservoirs['res_catch']=reservoirs['geometry'].map(lambda x: catch_shape['ncatch'].loc[catch_shape.contains(x)].values[0])
reservoirs['res_type']='reservoir'
reservoirs['wStorIni']=reservoirs['wStorCap']/2
reservoirs['wStorFin']=reservoirs['wStorCap']/2
reservoirs['wkV']=0
reservoirs['wResArea']=0
WIres=reservoirs	[['nres','res_type','res_catch','wStorCap','wStorIni','wStorFin','wkV','wResArea']]	
#export
export=os.path.join(folder_path,'WIformat_Reservoirs.csv')
WIres.set_index('nres').to_csv(export)

#%%Total runoff (to include upstream inflow - skip if directly available) (code fron hydroeconomic_optimization.py)
#RUN BEFORE HYDROPOWER
rpath=r'C:\Users\RAPY\OneDrive - COWI\WHAT_IF\Data\x_runoff_clm45_gfdl_esm2m_rcp26_jan2020.csv'
Q = pd.read_csv(rpath)
Q=Q.iloc[: , 1:] #remove first col with index
catch_shape['catch_ds']=catch_shape['TO_BAS'].apply(lambda k: 'c'+str(k)) #to get network
catch_ds={catch_shape['ncatch'].loc[k]:catch_shape['catch_ds'].loc[k] for k in catch_shape.index}
Qtot=Q #initialize
for t in Q.index: #time
    for c in Q.columns: #catchments
        Qtot.loc[t,c]=Q.loc[t,c]
        upcatch=[c]
        while len(upcatch)>0:
            UpInflow = sum(Q.loc[t,kc] for kc in Q.columns if catch_ds[kc] in upcatch)
            Qtot.loc[t,c] += UpInflow
            upcatch=[kc for kc in Q.columns if catch_ds[kc] in upcatch]
            
#%%Hydropower plants
#RUN SECTION ABOVE BEFORE
MINCAP=100 #minimum capacity to be considered (MW)
from shapely.geometry import Point
path = r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\Literature\hp_naturepaper\COPY_data_almeida_et_al_GEC.csv'
pdata = pd.read_csv(path,encoding="ISO-8859-1") #normally encoding not specified, but special encoding used here
geometry = [Point(xy) for xy in zip(pdata.longitude, pdata.latitude)]
gpdata = gpd.GeoDataFrame(pdata, crs=bCRS, geometry=geometry)
hp = gpd.clip(gpdata,gpd.read_file(basin_file).to_crs(bCRS))
#Associate ncatch to geometry
hp['hp_catch'] = hp['geometry'].map(lambda x:catch_shape['ncatch'].loc[catch_shape.contains(x)].values[0])
#Calculate mean flow and efficiency indexes
#Average flow
Qmax=Q.mean()*1.43 #based on nature paper-valid for amazon https://doi.org/10.1016/j.gloenvcha.2021.102383 
#Turbinable flow per year (assumption is everything below Qmax can be used)
flow=hp['hp_catch'].apply(lambda x: sum(max(q,Qmax.loc[x]) for q in Qtot.loc[:,x]))/(len(Q.index)/12)
#Production factor in kWh/m3 (or GWh/Mm3) = Production/Turbinable flow (including efficiency)
hp['eHppProd']=hp['EnergyProductionBaseline (GWh/year)']/flow
#Get country
cpath=r'C:\Users\rapy\OneDrive - COWI\Amazon_nexus\QGIS\Boundaries\amazon_countries_clipped.shp'
countries=gpd.read_file(cpath).to_crs(bCRS)
hp['hp_country'] = hp['geometry'].map(lambda x:countries['CNTRY_NAME'].loc[countries.contains(x)].values[0])
hp['hp_pmarket'] = hp['hp_country'].map(lambda x: 'PM_'+x)

#To WHAT-IF
hp.rename(columns={'Dam':'nhpp','Capacity_MW':'eHppCap'},inplace=True)
hp=hp[hp['eHppCap']>MINCAP]
hp['nhpp']=hp['nhpp'].map(remove_accent)
hp['eHppEff']=1
hp['eHppCost']=0.006
hp['eHppVal']=0
hp['wMaxTurb']=hp['hp_catch'].apply(lambda x: Qmax.loc[x])*10**6/(3600*24*365/12)
hp.set_index('nhpp',inplace=True)
#Hydropower to reservoirs
def hp_res(geom):
    res=reservoirs['nres'].loc[reservoirs.within(geom.buffer(0.15))].values
    return res[0] if len(res)>0 else 'ROR'
hp['hp_res'] = hp['geometry'].map(hp_res)
#Export
WIhp = hp[['hp_res','hp_pmarket','eHppProd','eHppCap','eHppCost','eHppEff',
           'eHppVal','hp_country','hp_catch','wMaxTurb','Status']]
hp.to_csv(os.path.join(folder_path,'HP_data_almeida_et_al_GEC.csv'))
WIhp.to_csv(os.path.join(folder_path,'WIformat_Hydropower.csv'))     

#%%Fish
#https://figshare.com/articles/dataset/A_database_of_freshwater_fish_species_of_the_Amazon_Basin/9923762
file=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\Water\fish\CompleteDatabase_b.csv'
file=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\Water\fish\fish_database_b.shp'
fishb=gpd.read_file(file).to_crs(bCRS)
joined=gpd.sjoin(fishb,catch_shape,how='left',op='within')
joined['nbfish']=1
a=joined.groupby(['ncatch','Referent.S']).sum()
a['nbspecies']=1
b=joined.groupby(['ncatch']).sum()
bb=a.groupby(['ncatch']).sum()
c=joined.groupby(['ncatch','Longitude.','Latitude.Y']).sum()
c['nbsamples']=1
cc=c.groupby('ncatch').sum()
catch_shape['']
catch_shape['nbsamples']=catch_shape['ncatch'].apply(lambda k: cc.loc[k]['nbsamples'] if k in cc.index else 0)
catch_shape['nbspecies']=catch_shape['ncatch'].apply(lambda k: bb.loc[k]['nbspecies'] if k in cc.index else 0)
catch_shape['nboccurence']=catch_shape['ncatch'].apply(lambda k: bb.loc[k]['nbfish'] if k in cc.index else 0)
catch_shape['s_p_sample']=catch_shape['nbspecies']/catch_shape['nbsamples']
catch_shape['o_p_sample']=catch_shape['nboccurence']/catch_shape['nbsamples']
#df = pd.read_csv(file,sep=';',decimal='.')
#geometry = [Point(xy) for xy in zip(df.x, df.y)]
#crs = {'init': 'epsg:2263'} #http://www.spatialreference.org/ref/epsg/2263/
#geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)    

#%% Indigenous territories
file=r'C:\Users\RAPY\OneDrive - COWI\Amazon_nexus\QGIS\Socioeconomic\Tis_2020\Tis_TerritoriosIndigenas.shp'
tis=gpd.read_file(file).to_crs(bCRS)
a=gpd.overlay(tis, catch_shape, how='union')
#%%            
#EXPORT
#exportname='output_'+cl_name+'.xlsx'

exportname='fish_data_by_catch'
EXPORT=1
if EXPORT==1:
    outpath=os.path.join(folder_path,exportname+'.xlsx') #path of excel file to dump results (does not need to exist)
    export_to_excel(outpath,catch_shape)
    catch_shape.to_file(os.path.join(folder_path,exportname+'.shp'))
    
#%% DOWNLOAD EXPERIMENT
#import urllib.request 
#import xarray as xr
# url = 'https://files.isimip.org/ISIMIP2b/OutputData/agriculture/PEPIC/gfdl-esm2m/future/pepic_gfdl-esm2m_ewembi_rcp60_2005soc_co2_yield-soy-noirr_global_annual_2006_2099.nc4'
# file = url.split('/future/')[1]
# if not os.path.exists(file):
#     urllib.request.urlretrieve(url, file)
# req = urllib.request.Request(url)
# with urllib.request.urlopen(req) as resp:
#     with rioxarray.open_rasterio(io.BytesIO(resp.read()),decode_times=False) as ncdata:
#         ncdata.rio.write_crs(bCRS, inplace=True)
#     #ds = xr.open_dataset(io.BytesIO(resp.read()),decode_times=False)