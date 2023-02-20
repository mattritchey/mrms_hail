# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 09:56:29 2022

@author: mritchey
"""
#streamlit run "C:\Users\mritchey\.spyder-py3\Python Scripts\streamlit projects\mrms\mrms_hail2 buffer.py"

import plotly.express as px
import os
from PIL import Image
from joblib import Parallel, delayed
import pandas as pd
import streamlit as st
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import math
import geopandas as gpd
from skimage.io import imread
from streamlit_plotly_events import plotly_events
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import rasterio
import rioxarray
import numpy as np


@st.cache
def geocode(address, buffer_size):
    try:
        address2 = address.replace(' ', '+').replace(',', '%2C')
        df = pd.read_json(
            f'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address={address2}&benchmark=2020&format=json')
        results = df.iloc[:1, 0][0][0]['coordinates']
        lat, lon = results['y'], results['x']
    except:
        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(address)
        lat, lon = location.latitude, location.longitude

    df = pd.DataFrame({'Lat': [lat], 'Lon': [lon]})
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Lon, df.Lat, crs=4326))
    gdf['buffer'] = gdf['geometry'].to_crs(
        3857).buffer(buffer_size/2*2580).to_crs(4326)
    return gdf


@st.cache
def get_pngs(date):
    year, month, day = date[:4], date[4:6], date[6:]
    url = f'https://mrms.nssl.noaa.gov/qvs/product_viewer/local/render_multi_domain_product_layer.php?mode=run&cpp_exec_dir=/home/metop/web/specific/opv/&web_resources_dir=/var/www/html/qvs/product_viewer/resources/&prod_root={prod_root}&qperate_pal_option=0&qpe_pal_option=0&year={year}&month={month}&day={day}&hour={hour}&minute={minute}&clon={lon}&clat={lat}&zoom={zoom}&width=920&height=630'
    data = imread(url)[:, :, :3]
    data2 = data.reshape(630*920, 3)
    data2_df = pd.DataFrame(data2, columns=['R', 'G', 'B'])
    data2_df2 = pd.merge(data2_df, lut[['R', 'G', 'B', 'Hail Scale', 'Hail Scale In']], on=['R', 'G', 'B'],
                         how='left')[['Hail Scale', 'Hail Scale In']]
    data2_df2['Date'] = date
    return data2_df2.reset_index()


@st.cache
def get_pngs_parallel(dates):
    results1 = Parallel(n_jobs=32, prefer="threads")(
        delayed(get_pngs)(i) for i in dates)
    return results1


@st.cache
def png_data(date):
    year, month, day = date[:4], date[4:6], date[6:]
    url = f'https://mrms.nssl.noaa.gov/qvs/product_viewer/local/render_multi_domain_product_layer.php?mode=run&cpp_exec_dir=/home/metop/web/specific/opv/&web_resources_dir=/var/www/html/qvs/product_viewer/resources/&prod_root={prod_root}&qperate_pal_option=0&qpe_pal_option=0&year={year}&month={month}&day={day}&hour={hour}&minute={minute}&clon={lon}&clat={lat}&zoom={zoom}&width=920&height=630'
    data = imread(url)
    return data


@st.cache(allow_output_mutation=True)
def map_folium(data, gdf):
    m = folium.Map(location=[lat, lon],  zoom_start=zoom, height=300)
    folium.Marker(
        location=[lat, lon],
        popup=address).add_to(m)

    folium.GeoJson(gdf['buffer']).add_to(m)
    folium.raster_layers.ImageOverlay(
        data, opacity=0.8, bounds=bounds).add_to(m)
    return m


def to_radians(degrees):
  return degrees * math.pi / 180


def lat_lon_to_bounds(lat, lng, zoom, width, height):
    earth_cir_m = 40075016.686
    degreesPerMeter = 360 / earth_cir_m
    m_pixel_ew = earth_cir_m / math.pow(2, zoom + 8)
    m_pixel_ns = earth_cir_m / \
        math.pow(2, zoom + 8) * math.cos(to_radians(lat))

    shift_m_ew = width/2 * m_pixel_ew
    shift_m_ns = height/2 * m_pixel_ns

    shift_deg_ew = shift_m_ew * degreesPerMeter
    shift_deg_ns = shift_m_ns * degreesPerMeter

    return [[lat-shift_deg_ns, lng-shift_deg_ew], [lat+shift_deg_ns, lng+shift_deg_ew]]


def image_to_geotiff(bounds, input_file_path, output_file_path='template.tiff'):
    south, west, north, east = tuple(
        [item for sublist in bounds for item in sublist])
    dataset = rasterio.open(input_file_path, 'r')
    bands = [1, 2, 3]
    data = dataset.read(bands)
    transform = rasterio.transform.from_bounds(west, south, east, north,
                                               height=data.shape[1],
                                               width=data.shape[2])
    crs = {'init': 'epsg:4326'}

    with rasterio.open(output_file_path, 'w', driver='GTiff',
                       height=data.shape[1],
                       width=data.shape[2],
                       count=3, dtype=data.dtype, nodata=0,
                       transform=transform, crs=crs,
                       compress='lzw') as dst:
        dst.write(data, indexes=bands)


def get_mask(bounds, buffer_size):
    year, month, day = date[:4], date[4:6], date[6:]
    url = f'https://mrms.nssl.noaa.gov/qvs/product_viewer/local/render_multi_domain_product_layer.php?mode=run&cpp_exec_dir=/home/metop/web/specific/opv/&web_resources_dir=/var/www/html/qvs/product_viewer/resources/&prod_root={prod_root}&qperate_pal_option=0&qpe_pal_option=0&year={year}&month={month}&day={day}&hour={hour}&minute={minute}&clon={lon}&clat={lat}&zoom={zoom}&width=920&height=630'
    img_data = requests.get(url, verify=False).content
    input_file_path = f'image_name_{date}_{var}.png'
    output_file_path = 'template.tiff'
    with open(input_file_path, 'wb') as handler:
        handler.write(img_data)

    image_to_geotiff(bounds, input_file_path, output_file_path)
    rds = rioxarray.open_rasterio(output_file_path)
    # rds.plot.imshow()

    rds = rds.assign_coords(distance=(haversine(rds.x, rds.y, lon, lat)))
    mask = rds['distance'].values <= buffer_size
    mask = np.transpose(np.stack([mask, mask, mask]), (1, 2, 0))
    return mask


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


#Set Columns
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns((3))
col1, col2, col3 = st.columns((3, 3, 1))

#Input Data
zoom = 10
address = st.sidebar.text_input(
    "Address", "123 Main Street, Cincinnati, OH 43215")

date = st.sidebar.date_input("Date",  pd.Timestamp(
    2022, 7, 6), key='date').strftime('%Y%m%d')
d = pd.Timestamp(date)
days_within = st.sidebar.selectbox('Within Days:', (5, 30, 60))
var = 'Hail'
var_input = 'hails&product=MESHMAX1440M'
mask_select = st.sidebar.radio('Only Show Buffer Data:', ("No", "Yes"))
buffer_size = st.sidebar.radio('Buffer Size (miles):', (5, 10, 15))

year, month, day = date[:4], date[4:6], date[6:]
hour = 23
minute = 30

prod_root = var_input[var_input.find('=')+1:]

#Geocode
gdf = geocode(address, buffer_size)
lat, lon = tuple(gdf[['Lat', 'Lon']].values[0])

#Get Value
url = 'https://mrms.nssl.noaa.gov/qvs/product_viewer/local/get_multi_domain_rect_binary_value.php?mode=run&cpp_exec_dir=/home/metop/web/specific/opv/&web_resources_dir=/var/www/html/qvs/product_viewer/resources/'\
    + f'&prod_root={prod_root}&lon={lon}&lat={lat}&year={year}&month={month}&day={day}&hour={hour}&minute={minute}'

response = requests.get(url, verify=False).json()
qvs_values = pd.DataFrame(response, index=[0])[
    ['qvs_value', 'qvs_units']].values[0]
qvs_value = qvs_values[0]
qvs_unit = qvs_values[1]

#Get PNG Focus
data = png_data(date)

#Legend
legend = Image.open('hail scale2.png')

#Get PNG Max
start_date, end_date = d - \
    pd.Timedelta(days=days_within), d+pd.Timedelta(days=days_within)
dates = pd.date_range(start_date,
                      end_date).strftime('%Y%m%d')
lut = pd.read_csv('hail scale.csv')
bounds = lat_lon_to_bounds(lat, lon, zoom, 920, 630)


results1 = get_pngs_parallel(dates)
# results1 = Parallel(n_jobs=32, prefer="threads")(delayed(get_pngs)(i) for i in dates)
results = pd.concat(results1)
max_data = results.groupby('index')[['Hail Scale']].max()

max_data2 = pd.merge(max_data,
                     lut[['R', 'G', 'B', 'Hail Scale']],
                     on=['Hail Scale'],
                     how='left')[['R', 'G', 'B']]

data_max = max_data2.values.reshape(630, 920, 3)

#Masked Data
if mask_select == "Yes":
    mask = get_mask(bounds, buffer_size)
    mask1 = mask[:, :, 0].reshape(630*920)
    results = pd.concat([i[mask1] for i in results1])
    data_max = data_max*mask
else:
    pass


#Bar
bar = results.query("`Hail Scale`>4").groupby(
    ['Date', 'Hail Scale In'])['index'].count().reset_index()
bar['Date'] = pd.to_datetime(bar['Date'])

bar = bar.reset_index()
bar.columns = ['level_0', 'Date', 'Hail Scale In', 'count']
bar['Hail Scale In'] = bar['Hail Scale In'].astype(str)
bar = bar.sort_values('Hail Scale In', ascending=True)

color_discrete_map = lut[['Hail Scale In', 'c_code']].sort_values(
    'Hail Scale In', ascending=True).astype(str)
color_discrete_map = color_discrete_map.set_index(
    'Hail Scale In').to_dict()['c_code']

fig = px.bar(bar, x="Date", y="count", color="Hail Scale In",
             barmode='stack',
             color_discrete_map=color_discrete_map)

#Submit Url to New Tab
url = f'https://mrms.nssl.noaa.gov/qvs/product_viewer/index.php?web_exec_mode=run&menu=menu_config.txt&year={year}&month={month}&day={day}&hour=23&minute=30&time_mode=static&zoom=9&clon={lon}&clat={lat}&base=0&overlays=1&mping_mode=0&product_type={var_input}&qpe_pal_option=0&opacity=.75&looping_active=off&num_frames=6&frame_step=200&seconds_step=600'


#Map Focus
m = map_folium(data, gdf)
#Map Max
m_max = map_folium(data_max, gdf)

with st.container():
    col1, col2, col3 = st.columns((1, 2, 2))
    with col1:
        link = f'[Go To MRMS Site]({url})'
        st.markdown(link, unsafe_allow_html=True)
        st.image(legend)
    with col2:
        st.header(f'{var} on {pd.Timestamp(date).strftime("%D")}')
        st_folium(m, height=300)
    with col3:
        st.header(
            f'Max from {start_date.strftime("%D")} to {end_date.strftime("%D")}')
        st_folium(m_max, height=300)

try:
    selected_points = plotly_events(fig, click_event=True, hover_event=False)
    date2 = pd.Timestamp(selected_points[0]['x']).strftime('%Y%m%d')
    data2 = png_data(date2)
    m3 = map_folium(data2, gdf)
    st.header(f'{var} on {pd.Timestamp(date2).strftime("%D")}')
    st_folium(m3, height=300)
except:
    pass


st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
