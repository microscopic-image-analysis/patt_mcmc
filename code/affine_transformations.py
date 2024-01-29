"""
Auxiliary module implementing six different types of affine transforms in a way
that allows for both efficient evaluation and pickling by multiprocessing etc.
"""

def affine_trf_trivial(y, cen_para, cov_para):
    return y

def affine_trf_cen(y, cen_para, cov_para):
    return y + cen_para

def affine_trf_var(y, cen_para, cov_para):
    return cov_para * y

def affine_trf_cen_var(y, cen_para, cov_para):
    return cov_para * y + cen_para

def affine_trf_cov(y, cen_para, cov_para):
    return cov_para @ y

def affine_trf_cen_cov(y, cen_para, cov_para):
    return cov_para @ y + cen_para

class affine_trf:
    """Auxiliary class to deal with the issue that multiprocessing refuses to 
        pickle local functions; defines pickle-able affine transformations
    """
    def __init__(self, cen_para, cov_para):
        self.cen_para = cen_para
        self.cov_para = cov_para
        if type(self.cov_para) == type(None):
            if type(self.cen_para) == type(None):
                self.eval = affine_trf_trivial
            else:
                self.eval = affine_trf_cen
        elif len(self.cov_para.shape) == 1:
            if type(self.cen_para) == type(None):
                self.eval = affine_trf_var
            else:
                self.eval = affine_trf_cen_var
        elif len(self.cov_para.shape) == 2:
            if type(self.cen_para) == type(None):
                self.eval = affine_trf_cov
            else:
                self.eval = affine_trf_cen_cov
    def __call__(self, y):
        return self.eval(y, self.cen_para, self.cov_para)


def inv_affine_trf_trivial(x, cen_para, inv_cov_para):
    return x

def inv_affine_trf_cen(x, cen_para, inv_cov_para):
    return x - cen_para

def inv_affine_trf_var(x, cen_para, inv_cov_para):
    return inv_cov_para * x

def inv_affine_trf_cen_var(x, cen_para, inv_cov_para):
    return inv_cov_para * (x - cen_para)

def inv_affine_trf_cov(x, cen_para, inv_cov_para):
    return inv_cov_para @ x

def inv_affine_trf_cen_cov(x, cen_para, inv_cov_para):
    return inv_cov_para @ (x - cen_para)

class inv_affine_trf:
    """Auxiliary class to deal with the issue that multiprocessing refuses to 
        pickle local functions; defines pickle-able affine transformations
    """
    def __init__(self, cen_para, inv_cov_para):
        self.cen_para = cen_para
        self.inv_cov_para = inv_cov_para
        if type(self.inv_cov_para) == type(None):
            if type(self.cen_para) == type(None):
                self.eval = inv_affine_trf_trivial
            else:
                self.eval = inv_affine_trf_cen
        elif len(self.inv_cov_para.shape) == 1:
            if type(self.cen_para) == type(None):
                self.eval = inv_affine_trf_var
            else:
                self.eval = inv_affine_trf_cen_var
        elif len(self.inv_cov_para.shape) == 2:
            if type(self.cen_para) == type(None):
                self.eval = inv_affine_trf_cov
            else:
                self.eval = inv_affine_trf_cen_cov
    def __call__(self, x):
        return self.eval(x, self.cen_para, self.inv_cov_para)

