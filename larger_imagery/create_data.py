import shutil
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import matplotlib
from MakeDirs import MakeDirs
import pyproj
import ray
import sys
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
import geopandas
import pandas as pd
from satpy import Scene
from PIL import Image, ImageOps
import os
import random
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import s3fs
import pytz
import multiprocessing
import shutil
import wget
from suntime import Sun
from datetime import timedelta

def get_indices():
    file_list = glob.glob('{}truth/*/*/*'.format(dn_dir))
    indices = []
    for fn in file_list:
        idx = fn.split('_')[-1].split('.')[0]
        indices.append(idx)
    indices = list(set(indices))
    indices.sort()
    return indices

def mv_files(truth_srcs, yr_dn, idx):
    print(truth_srcs)
    for truth_src in truth_srcs:
        coords_src = truth_src.replace('truth','coords')
        data_src = truth_src.replace('truth','data')
        truth_dst = truth_src.replace('/temp_data/{}'.format(yr_dn),'')
        coords_dst = coords_src.replace('/temp_data/{}'.format(yr_dn),'')
        data_dst = data_src.replace('/temp_data/{}'.format(yr_dn),'')
        shutil.copyfile(truth_src, truth_dst)
        shutil.copyfile(coords_src, coords_dst)
        shutil.copyfile(data_src, data_dst)

def find_best_data(yr, dn):
    indices = get_indices()
    for idx in indices:
        truth_file_list = glob.glob('{}truth/*/*/*_{}.tif'.format(dn_dir, idx))
        if len(truth_file_list) == 2:
            yr_dn = yr+dn
            mv_files(truth_file_list, yr_dn, idx)
        else:
            print(truth_file_list)
            print('there arent two files!')

def get_start_end_from_fn(fn):
    start = fn.split('_')[1][1:-1]
    fn_start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    return pytz.utc.localize(fn_start_dt), pytz.utc.localize(fn_start_dt)

def file_exists(yr, fn_heads, idx, density):
    data_dst = "/scratch/alpine/mecr8410/both_sats/data/"
    data_loc = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/"
    for fn_head in fn_heads:
        dst_file = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(data_dst, yr, density, fn_head, idx))
        if len(dst_file) > 0:
            print('YOUVE ALREADY MADE THAT FILE:', dst_file, flush=True)
            return None, None

        file_list = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(data_loc, yr, density, fn_head, idx))
        if len(file_list) > 0:
            print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
            fn = file_list[0].split('/')[-1]
            start_dt, end_dt = get_start_end_from_fn(fn)
            return start_dt, end_dt
        file_list = glob.glob('{}low_iou/{}_{}.tif'.format(data_loc, fn_head, idx))
        if len(file_list) > 0:
            print("THIS ANNOTATION FAILED:", file_list[0], flush=True)
            return None, None
    return  None, None

def check_bounds(x, y, bounds):
    if bounds['minx'] > np.min(x) and bounds['maxx'] < np.max(x) and bounds['miny'] > np.min(y) and bounds['maxy'] < np.max(y):
        return True
    else:
        return False

def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        end = row['End']
        start = row['Start']
        # the ranges overlap if:
        if t_0-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= t_f:
            use_idx.append(idx)
    rel_smoke = smoke_shape.loc[use_idx]
    return rel_smoke

def reshape(A, idx, size=256):
    print('before reshape: ', np.sum(A))
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    print('after reshape: ', np.sum(A))
    return A

def save_data(R, G, B, idx, fn_data, size=256):
    R = reshape(R, idx, size)
    G = reshape(G, idx, size)
    B = reshape(B, idx, size)
    layers = np.dstack([R, G, B])
    total = np.sum(R) + np.sum(G) + np.sum(B)
    if total > 4e5 and total < 1e15:
        skimage.io.imsave(fn_data, layers)
        return True
    return False

def get_rand_center(idx, rand_xy):
    x_o = idx[0] + rand_xy[0]
    y_o = idx[1] + rand_xy[1]
    return (x_o, y_o)

def find_closest_pt(pt_x, pt_y, x, y):
    x_diff = np.abs(x - pt_x)
    y_diff = np.abs(y - pt_y)
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    sum_diff = x_diff2 + y_diff2
    dist = sum_diff**(1/2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    #if distance is less than 1km away
    if np.min(dist) < 1000:
        return idx
    else:
        print("not close enough")
        return None

def get_centroid(center, x, y, img_shape, rand_xy):
    pt_x = center.x
    pt_y = center.y
    idx = find_closest_pt(pt_x, pt_y, x, y)
    if idx:
        rand_idx = get_rand_center(idx, rand_xy)
        return idx, rand_idx
    else:
        return None, None
def plot_coords(lat, lon, idx, tif_fn):
    lat_coords = reshape(lat, idx)
    lon_coords = reshape(lon, idx)
    coords_layers = np.dstack([lat_coords, lon_coords])
    skimage.io.imsave(tif_fn, coords_layers)

def plot_truth(x, y, lcc_proj, smoke, png_fn, idx, img_shape):
    fig = plt.figure(figsize=(img_shape[1]/100, img_shape[0]/100), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    smoke.plot(ax=ax, facecolor='black')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(png_fn, dpi=100)
    plt.close(fig)
    img = Image.open(png_fn)
    bw = img.convert('1')
    bw = ImageOps.invert(bw)

    truth = np.asarray(bw).astype('i')
    truth = reshape(truth, idx)
    os.remove(png_fn)
    return truth

def get_truth(x, y, lcc_proj, smoke, idx, png_fn, tif_fn, center, img_shape):

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    # high = [1,1,1], med = [0, 1, 1], low = [0, 0, 1]
    low_truth = plot_truth(x, y, lcc_proj, low_smoke, png_fn, idx, img_shape)
    med_truth = plot_truth(x, y, lcc_proj, med_smoke, png_fn, idx, img_shape)
    high_truth = plot_truth(x, y, lcc_proj, high_smoke, png_fn, idx, img_shape)
    low_truth += med_truth + high_truth
    low_truth = np.clip(low_truth, 0, 1)
    med_truth += high_truth
    med_truth = np.clip(med_truth, 0, 1)

    truth_layers = np.dstack([high_truth, med_truth, low_truth])
    print('---------------------------')
    print(tif_fn)
    print(np.sum(truth_layers))
    print('---------------------------')
    if np.sum(truth_layers) > 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    return False

def get_extent(center):
    x0 = center.x - 2.0e5
    y0 = center.y - 2.0e5
    x1 = center.x + 2.0e5
    y1 = center.y + 2.0e5
    return [x0, y0, x1, y1]

def get_scn(fns, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)
    scn.load(['C01', 'C02', 'C03'], generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=1000,
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return new_scn

def get_get_scn(sat_fns, extent, sleep_time=0):
    time.sleep(sleep_time)
    scn = get_scn(sat_fns, extent)
    return scn

def create_data_truth(sat_fns, smoke, idx0, yr, density, rand_xy):
    print('idx: ', idx0)
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.CRS.from_user_input(lcc_str)
    smoke_lcc = smoke.to_crs(lcc_proj)
    centers = smoke_lcc.centroid
    center = centers.loc[idx0]
    try:
        extent = get_extent(center)
    except:
        return fn_head

    try:
        scn = get_get_scn(sat_fns, extent)
    except:
        try:
            print('wait 120 seconds')
            scn = get_get_scn(sat_fns, extent, 120)
        except:
            print('{} wouldnt download, moving on'.format(sat_fns[0]))
            return fn_head

    composite = 'C01'
    lcc_proj = scn[composite].attrs['area'].to_cartopy_crs()
    smoke_lcc = smoke.to_crs(lcc_proj)
    scan_start = pytz.utc.localize(scn[composite].attrs['start_time'])
    scan_end = pytz.utc.localize(scn[composite].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    x = scn[composite].coords['x']
    y = scn[composite].coords['y']
    lon, lat = scn[composite].attrs['area'].get_lonlats()
    img_shape = scn[composite].shape

    R = np.squeeze(scn['C02'].data.compute())
    R[np.isnan(R)] = 0
    G = np.squeeze(scn['C03'].data.compute())
    G[np.isnan(G)] = 0
    B = np.squeeze(scn['C01'].data.compute())
    B[np.isnan(B)] = 0

    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T

    cent, idx = get_centroid(center, xx, yy, img_shape, rand_xy)

    if cent:
        png_fn_truth = dn_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
        tif_fn_truth = dn_dir + 'truth/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        print(tif_fn_truth)
        tif_fn_data = dn_dir + 'data/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_coords = dn_dir + 'coords/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        data_saved = save_data(R, G, B, idx, tif_fn_data)
        if data_saved:
            truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, idx, png_fn_truth, tif_fn_truth, center, img_shape)
            if truth_saved:
                plot_coords(lat, lon, idx, tif_fn_coords)
    return fn_head

def get_closest_file(fns, best_time, sat_num):
    diff = timedelta(days=100)
    use_fns = []
    for fn in fns:
        starts = []
        if 'C01' in fn:
            s_e = fn.split('_')[3:5]
            start = s_e[0]
            end = s_e[1][0:11]
            C02_fn = 'C02_G{}_{}_{}'.format(sat_num, start, end)
            C03_fn = 'C03_G{}_{}_{}'.format(sat_num, start, end)
            for f in fns:
                if C02_fn in f:
                   C02_fn = f
                elif C03_fn in f:
                   C03_fn = f
            if 'nc' in C02_fn and 'nc' in C03_fn:
                start = s_e[0][1:-3]
                s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
                if diff > abs(s_dt - best_time):
                    diff = abs(s_dt - best_time)
                    use_fns = [fn, C02_fn, C03_fn]
    return use_fns

def get_smoke(yr, month, day):
    smoke_dir = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/smoke/"
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    print('DOWNLOADING SMOKE:')
    print(fn)
    smoke_shape_fn = smoke_dir + 'hms_smoke{}{}{}.shp'.format(yr,month,day)
    if os.path.exists(smoke_dir+fn):
        print("{} already exists".format(fn))
        smoke = geopandas.read_file(smoke_shape_fn)
        return smoke
    else:
        try:
            url = 'https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{}/{}/{}'.format(yr, month, fn)
            filename = wget.download(url, out=smoke_dir)
            shutil.unpack_archive(filename, smoke_dir)
            smoke = geopandas.read_file(smoke_shape_fn)
            return smoke
        except Exception as e:
            print(e)
            print('NO SMOKE DATA FOR THIS DATE')
            return None

def get_sat_files(idx, bounds, density, s_dt, e_dt):
    fs = s3fs.S3FileSystem(anon=True)
    tt = s_dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)
    all_fn_heads = []
    all_sat_fns = []
    t = s_dt
    time_list = [t]
    while t < e_dt:
        t += timedelta(minutes=10)
        time_list.append(t)
    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if s_dt >= G18_op_dt:
        sat_nums = ['18', '16']
    elif s_dt >= G17_op_dt:
        sat_nums = ['17', '16']
    else:
        print("G17 not launched yet")
        return None, None
    for sat_num in sat_nums:
        for curr_time in time_list:
            hr = curr_time.hour
            hr = str(hr).zfill(2)
            yr = curr_time.year
            dn = curr_time.strftime('%j')
            full_filelist = []
            view = 'F'
            try:
                full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
            except Exception as e:
                print("ERROR WITH FS LS")
                print(sat_num, view, yr, dn, hr)
                print(e)
            if len(full_filelist) == 0 and sat_num == '18' and curr_time < G17_end_dt:
                sat_num = '17'
                try:
                    full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
                except Exception as e:
                    print("ERROR WITH FS LS")
                    print(sat_num, view, yr, dn, hr)
                    print(e)
            if len(full_filelist) > 0:
                sat_fns = get_closest_file(full_filelist, curr_time, sat_num)
                if sat_fns:
                    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
                    all_fn_heads.append(fn_head)
                    all_sat_fns.append(sat_fns)
    if len(all_sat_fns)>0:
        all_sat_fns = [list(item) for item in set(tuple(row) for row in all_sat_fns)]
        all_fn_heads = list(set(all_fn_heads))
        return all_fn_heads, all_sat_fns
    return None, None

def get_file_locations(use_fns):
    file_locs = []
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = dn_dir + 'goes_temp/'
    for file_path in use_fns:
        fn = file_path.split('/')[-1]
        dl_loc = goes_dir+fn
        file_locs.append(dl_loc)
        if os.path.exists(dl_loc):
            print("{} already exists".format(fn))
        else:
            print('downloading {}'.format(fn))
            fs.get(file_path, dl_loc)
    return file_locs

@ray.remote
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    yr = smoke_row['Start'].strftime('%Y')
    file_locs = smoke_row['file_locs']
    rand_xy = smoke_row['rand_xy']

    if len(file_locs) > 0:
        fns = create_data_truth(file_locs, smoke, idx, yr, density, rand_xy)
        return fns
    else:
        print('ERROR NO FILES FOUND FOR start time: ', smoke_row['Start'])

def run_no_ray(smoke_rows):
    fn_heads = []
    for smoke_row in smoke_rows:
        fn_head = iter_rows(smoke_row)
        fn_heads.append(fn_head)
    return fn_heads

def run_remote(smoke_rows):
    try:
        fn_heads = ray.get([iter_rows.remote(smoke_row) for smoke_row in smoke_rows])
        return fn_heads
    except Exception as e:
        print("ERROR WITH RAY GET")
        print(e)
        print(smoke_rows)
        fn_heads = []
        for smoke_row in smoke_rows:
            sat_fns = smoke_row['sat_fns']
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            fn_heads.append(fn_head)
        return fn_heads

# we need a consistent random shift in the dataset per each annotation
def get_random_xy(size=256):
    d = int(size/4)
    x_shift = random.randint(int(-1*d), d)
    y_shift = random.randint(int(-1*d), d)
    return (x_shift, y_shift)

def check_times_are_close(fn_heads, thresh):
    G16_dt = pytz.utc.localize(datetime.strptime(fn_heads[0].split('_')[1][:12], 's%Y%j%H%M'))
    G17_dt = pytz.utc.localize(datetime.strptime(fn_heads[1].split('_')[1][:12], 's%Y%j%H%M'))
    if abs(G16_dt - G17_dt) > thresh:
        print('-----------------')
        print("TIMES ARE TOO FAR!!!")
        print(fn_heads)
        print('-----------------')
        return False
    return True


def get_smoke_rows(smoke_rows, idx, smoke, density, rand_xy):
    ts_start = smoke.loc[idx]['Start']
    ts_end = smoke.loc[idx]['End']
    row_yr = ts_end.strftime('%Y')
    row_bounds = smoke.bounds.loc[idx]
    fn_heads, sat_fns = get_sat_files(idx, row_bounds, density, ts_start, ts_end)
    if sat_fns:
        start_dt, end_dt = file_exists(row_yr, fn_heads, idx, density)
        if start_dt:
            fn_heads, sat_fns = get_sat_files(idx, row_bounds, density, start_dt, end_dt)
            if len(fn_heads) == 2:
                if check_times_are_close(fn_heads, timedelta(minutes=20)):
                    for sat_fn_entry in sat_fns:
                        file_locs = get_file_locations(sat_fn_entry)
                        smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': row_bounds, 'density': density, 'sat_fns': sat_fn_entry, 'Start': ts_start, 'rand_xy': rand_xy, 'file_locs': file_locs}
                        smoke_rows.append(smoke_row)
    return smoke_rows

def smoke_utc(time_str):
    fmt = '%Y%j %H%M'
    return pytz.utc.localize(datetime.strptime(time_str, fmt))

# create object that contians all the smoke information needed
def create_smoke_row_dict(smoke):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    smoke_rows = []
    smoke_lcc = smoke.to_crs(3857)
    smoke_lcc_area = smoke_lcc['geometry'].area
    print('num rows: ', len(smoke))
    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)

    for idx, row in smoke.iterrows():
        rand_xy = get_random_xy()
        density = row['Density']
        if density == 'Medium' or density == 'Heavy':
            if smoke_lcc_area.loc[idx] > 1e7 and smoke_lcc_area.loc[idx] < 4e11:
                print('high or med idx:', idx)
                print("density area:", smoke_lcc.loc[idx]['geometry'].area)
                smoke_rows = get_smoke_rows(smoke_rows, idx, smoke, density, rand_xy)
            else:
                print('high or med idx:', idx)
                print('too high or low area:', smoke_lcc_area.loc[idx])

        elif density == 'Light' and smoke_lcc_area.loc[idx] > 1e8 and smoke_lcc_area.loc[idx] < 4e10:
            print('light idx:', idx)
            print("density area:", smoke_lcc_area.loc[idx])
            smoke_rows = get_smoke_rows(smoke_rows, idx, smoke, density, rand_xy)
        else:
            print('wrong sized light idx:', idx)
            print('too high or low area:', smoke_lcc_area.loc[idx])
    return smoke_rows

# remove large satellite files and the tif files created during corrections
def remove_files(fn_heads):
    fn_heads = list(set(fn_heads))
    print("REMOVING FILES")
    print(fn_heads)
    for head in fn_heads:
        for fn in glob.glob(dn_dir + 'goes_temp/*{}*'.format(head)):
            os.remove(fn)

# analysts can only label data that is taken during the daytime, we want to filter for geos data that was within the timeframe the analysts are looking at
def iter_smoke(date):

    dn = date[0]
    yr = date[1]
    s = '{}/{}'.format(yr, dn)
    fmt = '%Y/%j'
    dt = pytz.utc.localize(datetime.strptime(s, fmt))
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    print('------')
    print(dt)
    print('------')
    smoke = get_smoke(yr, month, day)

    if smoke is not None:
        smoke_rows = create_smoke_row_dict(smoke)
        ray_dir = "/scratch/alpine/mecr8410/tmp/{}{}".format(yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=8, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1')
        fn_heads = run_remote(smoke_rows)
        ray.shutdown()
        shutil.rmtree(ray_dir)
        #fn_heads = run_no_ray(smoke_rows)
        find_best_data(yr, dn)
        if fn_heads:
            remove_files(fn_heads)


def main2(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = '/scratch/alpine/mecr8410/both_sats/data/temp_data/{}{}/'.format(date[1], date[0])
        start = time.time()
        find_best_data(yr, dn)
        print(date)
        #iter_smoke(date)
        #shutil.rmtree(dn_dir)

def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    print(dns)
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    print(dates)
    for date in dates:
        print(date)
        dn_dir = '/scratch/alpine/mecr8410/both_sats/data/temp_data/{}{}/'.format(date[1], date[0])
        print(dn_dir)

        if not os.path.isdir(dn_dir):
            os.mkdir(dn_dir)
            MakeDirs(dn_dir, yr)
        start = time.time()
        iter_smoke(date)
        shutil.rmtree(dn_dir)
        print("Time elapsed for day {}: {}s".format(date, int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
