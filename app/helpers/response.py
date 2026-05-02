def success(message=None, code=200, data=None):
    return {"code": code, "message": message, "data": data}

def error(message=None, code=400, data=None):
    return {"code": code, "message": message, "data": data}

def info(message=None, code=200, data=None):
    return {"code": code, "message": message, "data": data}
