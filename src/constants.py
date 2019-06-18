
class VIEWS:
    L_CC = "L-CC"
    R_CC = "R-CC"
    L_MLO = "L-MLO"
    R_MLO = "R-MLO"

    LIST = [L_CC, R_CC, L_MLO, R_MLO]

    @classmethod
    def is_cc(cls, view):
        return view in (cls.L_CC, cls.R_CC)

    @classmethod
    def is_mlo(cls, view):
        return view in (cls.L_MLO, cls.R_MLO)

    @classmethod
    def is_left(cls, view):
        return view in (cls.L_CC, cls.L_MLO)

    @classmethod
    def is_right(cls, view):
        return view in (cls.R_CC, cls.R_MLO)


class VIEWANGLES:
    CC = "CC"
    MLO = "MLO"

    LIST = [CC, MLO]


class LABELS:
    LEFT_BENIGN = "left_benign"
    RIGHT_BENIGN = "right_benign"
    LEFT_MALIGNANT = "left_malignant"
    RIGHT_MALIGNANT = "right_malignant"

    LIST = [LEFT_BENIGN, RIGHT_BENIGN, LEFT_MALIGNANT, RIGHT_MALIGNANT]


INPUT_SIZE_DICT = {
    VIEWS.L_CC: (2677, 1942),
    VIEWS.R_CC: (2677, 1942),
    VIEWS.L_MLO: (2974, 1748),
    VIEWS.R_MLO: (2974, 1748),
}
