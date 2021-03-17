import osr

def epsg2wkt(in_proj):
    # 定义一个投影系统
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.SetFromUserInput(in_proj)
    wkt = inSpatialRef.ExportToWkt()
    return wkt