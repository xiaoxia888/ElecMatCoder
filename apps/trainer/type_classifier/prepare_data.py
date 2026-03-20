"""
TYPE 字段分类模型训练数据准备脚本
"""
import json
import random
from pathlib import Path
from typing import Dict, List

# TYPE 编码映射数据（与 seq2seq 相同）
TYPE_MAPPING = {
    "20ZN": ["20+Zn"],
    "45EL": ["45 DEGREE ELBOW", "45度弯头", "45°弯头", "45 ELBOW", "45°ELBOW", "无缝弯头\\45°", "有缝弯头\\45°", "45Deg Elb LR", "45Deg Elb"],
    "45ELW": ["45°焊接弯头"],
    "45ES": ["45° ELBOW SR"],
    "45HC": ["半管接头（带45°坡口）", "HALF COUPLING (WITH 45° BEVEL)"],
    "45LT": ["45度等径斜三通", "45度斜三通", "45度 Lateral Tee", "45度 Equal Tee", "45° Equal Tee", "45°等径斜三通", "等径45°斜三通", "45Deg Lateral"],
    "45O": ["45° WELDOLET", "45°对焊斜支管台"],
    "45RLT": ["45 度异径斜三通", "45度异径斜三通", "异径45°斜三通", "45度 Lateral Tee，异径", "45度Lateral Tee，异径", "45° Lateral Tee，异径", "45°Lateral Tee，异径", "45°异径斜三通"],
    "45RT": ["45° REDUCING TEE", "45°异径三通"],
    "45T": ["45°等径三通", "45° STRAIGHT TEE"],
    "60ELW": ["60°焊接弯头"],
    "8BF": ["SPECTACLE BLANK", "FIGURE 8", "8字盲板", "FIGURE-8", "SPECTACLEBLANK", "FIGURE8", "SpectacleBlind", "paddle blind"],
    "90EL": ["90 DEGREE ELBOW", "ELBOW(90-deg)", "90度弯头", "90°弯头", "90 ELBOW", "90°长半径弯头", "90度长半径弯头", "90ELBOW", "90° ELBOW", "90DEGREEELBOW", "90°ELBOW", "ELBOW 90°", "ELBOW, 90 DEG", "90°LR ELBOW", "ELBOW,90 DEG", "90° ELBOW LR", "无缝弯头\\90°", "有缝弯头\\90°", "90DEG ELBOW", "90Deg Elb LR", "90Deg Elb", "90EL"],
    "90EL3D": ["90 度弯头 3D", "90 DEGREE ELBOW 3D", "90 ELBOW 3D", "90ELBOW 3D"],
    "90EL6D": ["90 度弯头 6D", "90 DEGREE ELBOW 6D", "90 ELBOW 6D"],
    "90ELR": ["90°异径弯头", "Reducing Elbow"],
    "90ELT": ["90°螺纹弯头", "Elbow,THD-90E"],
    "90ES": ["90 度弯头 1D", "短半径弯头90ES", "90度弯头 1D", "90度弯头1D,", "90度弯头1D", "90 S.R. ELBOW", "90°弯头|Elbow,ES", "90° ELBOW SR", "90°短半径弯头", "90度短半径弯头", "90Deg Elb SR", "90DEGREEELBOW 1D", "90 DEGREE ELBOW 1D", "90 ELBOW 1D", "90ELBOW 1D", "90-DEG ELBOW 1D", "90EL;1D;SMLS;BE", "90EL;1D;SMLS", "90EL;1D;BE", "90度弯头;SR"],
    "90ESW": ["焊接90°弯头", "Elbow,W90ES"],
    "ACV": ["ANG CONTROL VALVE", "角控制阀", "ANGCONTROLVALVE"],
    "AV": ["ANGLE VALVE", "角阀", "ANGLEVALVE"],
    "B": ["Bolts and Nuts", "螺栓/螺母", "BOLT", "全螺纹螺柱&2个重型六角头螺母", "CONTINUOUS THREAD STUD", "全螺纹螺栓", "全螺纹螺柱", "SBlt"],
    "BE": ["BEND"],
    "BF": ["BLIND FLANGE", "BLD FLANGE", "盲法兰", "盲板", "法兰盖", "BLINDFLANGE", "BLDFLANGE", "FLANGE BL", "Bld Flg", "BF"],
    "BFV": ["Butterfly Valve", "蝶阀", "ButterflyValve", "But Waf"],
    "BST": ["BASKET TYPESTRAINER", "篮式过滤器", "BASKETTYPESTRAINER"],
    "BTV": ["Breathing Valve", "呼吸阀", "BreathingValve"],
    "BV": ["Ball Valve", "球阀", "BallValve", "Bal Flg"],
    "C": ["管接头", "活接头;NPT"],
    "CAP": ["cap", "管帽", "锻制管帽", "Cap", "TUBE CAP"],
    "CHV": ["CHECK VAVLE", "止回阀"],
    "CMH": ["波纹金属软管"],
    "CP": ["coupling", "管箍"],
    "CPC": ["COUPLING COLLAR TYPE", "松套法兰套圈", "COUPLING COLLAR"],
    "CROS": ["Special Cross"],
    "CST": ["CONE TYPE  STRAINER", "锥形过滤器"],
    "CTV": ["CONTROL VALVE", "控制阀", "调节阀"],
    "DCHV": ["DuPlChk Waf", "双板式止回阀"],
    "DCP": ["双头管箍"],
    "DV": ["Depressed valve", "pressure reducing valve", "减压阀"],
    "F": ["FLANGE", "WELDING NECK FLANGE", "WN Flange", "法兰", "WN FLANGE", "PAD FLANGE", "带颈对焊法兰", "FLANGE WN", "WN Flg"],
    "F45EL": ["FLANGED 45° ELBOW", "法兰45°弯头"],
    "F90EL": ["90度法兰弯头", "FLANGED 90° ELBOW", "法兰90°弯头"],
    "FA": ["阻火器"],
    "FE": ["孔板流量"],
    "FFN": ["FNPT螺纹法兰"],
    "FI": ["整体法兰"],
    "FJ": ["夹套法兰"],
    "FJDL": ["Jacketed Flange,Double-lip"],
    "FLM": ["FLOWMETER", "流量计"],
    "FN": ["THREAD FLANGE(NPT)", "THREADFLANGE(NPT)"],
    "FP": ["法兰管", "FLANGED PIPE"],
    "FPL": ["板式平焊法兰", "板式平焊不锈钢管法兰"],
    "FRC": ["FLANGED CONCENTRIC REDUCER", "法兰同心异径管", "RC;FLANGED", "同心异径管;FLANGED"],
    "FRCROS": ["FLANGED REDUCING CROSS"],
    "FRE": ["FLANGED ECCENTRIC REDUCER", "偏心异径管;FLANGED", "RE;FLANGED"],
    "FRT": ["法兰异径三通", "FLANGED REDUCING TEE", "RT;FLANGED"],
    "FSO": ["带颈平焊法兰", "SO FLANGE", "FLANGE SO"],
    "FT": ["法兰等径三通", "FLANGED STRAIGHT TEE"],
    "FTH": ["螺纹法兰", "螺纹法兰|Threaded Flange", "螺纹钢制管法兰"],
    "G": ["Gasket", "垫片"],
    "GLV": ["Globe Valve", "截止阀", "Glo Flg"],
    "GSV": ["globe stop valve", "球形截止阀"],
    "GV": ["Gate Valve", "闸阀", "Gat Flg"],
    "H": ["软管接头"],
    "HC": ["半管接头", "HOSECONNECTION", "HALF COUPLING", "Half Coup", "半耦合"],
    "HDC": ["HEAVY DUTY CLAMP"],
    "HPL": ["六角头管塞", "HEX HEAD PLUG"],
    "IT": ["INSTRUMENT TEE", "仪表三通"],
    "IV": ["INSTRUMENT VALVE", "仪表阀"],
    "JBP": ["夹套管隔板"],
    "JEP": ["夹套管端板"],
    "JSG": ["夹套管道视镜"],
    "KGV": ["KNIFE GATE VALVE"],
    "LBV": ["LINED BALL VALVE"],
    "LCHV": ["衬里止回阀", "LINED CHECK VALVE"],
    "LF": ["带颈对焊环松套法兰", "LJ Flg", "FLANGE LAPPED", "松套法兰"],
    "LFE": ["LONG FERRULE"],
    "LFTCHV": ["升降式止回阀", "LftChk Flg"],
    "LO": ["LATROLET"],
    "LP": ["钢衬管"],
    "LT": ["Lateral Tee"],
    "LWO": ["LIGHT WEIGHT OLET"],
    "MH": ["金属软管"],
    "N": ["nipple", "短节", "Nip", "短管接头", "PIPE NIPPLE"],
    "NMG": ["非金属平垫片", "NON-METALLIC FLAT GASKET", "NM Flat Gk", "NM Jk Flat Gk"],
    "NRV": ["non-return valve"],
    "NT": ["THREADED ONE END NIPPLE", "THREADEDONEENDNIPPLE"],
    "NV": ["NEEDLE VALVE", "针形阀", "NEEDLEVALVE", "针型阀"],
    "O": ["olet", "weldolet", "支管台", "对焊支管台", "对焊管接台", "OLET", "加强管接头", "Weldolet", "焊接支管台", "WELDING OUTLET", "对焊支管座", "不锈钢管件管接台"],
    "OB": ["Boss"],
    "OH": ["不补强板开口焊"],
    "ORG": ["金属环形垫（八角型）", "METALLIC RING-JOINT GASKET (OCTAGONAL)"],
    "OS": ["sockolet", "SOCKET OLET", "承插焊支管台", "承插焊管接台", "SOCKET OUTLET", "承插焊支管座", "承插支管座", "不锈钢管件承插焊管接台"],
    "OT": ["THREDOLET", "threadolet", "螺纹支管台", "螺纹管接台", "THREAD OUTLET"],
    "OW": ["开口焊", "补强板开口焊"],
    "P": ["pipe", "管子", "钢管", "PIPE", "Pipe", "TUBE PE", "管子;SMLS;PE", "TUBEPE", "不锈钢无缝管"],
    "P3PE": ["管子, 3PE加强级"],
    "PBV": ["PNEUMATIC BALL VALVE", "PNEUMATICBALLVALVE"],
    "PCV": ["Pressure Control Valve", "压力调节阀", "PressureControlValve"],
    "PI": ["仪表压力计"],
    "PJ": ["夹套钢管", "夹套钢管|Jacketed Pipe"],
    "PL": ["圆形丝堵", "管塞"],
    "PLV": ["PLUG VALVE", "旋塞阀"],
    "PU": ["PIPE UNION"],
    "PV": ["Pneumatic on-off Valve", "气动切断阀", "Pneumaticon-offValve"],
    "QR": ["Quick Release Coupling", "快速接头", "活接头"],
    "RC": ["swage nipple", "Concentric Swaged Nipple", "reducing coupling", "concentric reducer", "REDUCER(RC)", "同心异径管", "同心大小头", "同心管大小头", "CON. REDUCER", "同心异径", "异径管接头", "CON SWAGE", "CON REDUCER", "swagenipple", "ConcentricSwagedNipple", "reducingcoupling", "concentricreducer", "CON.REDUCER", "CONSWAGE", "CONREDUCER", "ReducerCon", "同心异径管接头", "CON. SWAGE", "CONC REDUCER", "CONCREDUCER", "CONC. REDUCER", "Red Coup", "Conc Swage", "CON SWAGE NIPPLE", "SWAGE NIPPLE CON", "RC", "同心异径短接"],
    "RCS": ["同心大小头(SW)"],
    "RDCP": ["同心异径双口管箍"],
    "RDCPN": ["双螺口异径管箍(NPT)"],
    "RE": ["Eccentric Swaged Nippl", "eccentric reduce", "REDUCER(RE)", "偏心异径管", "偏心大小头", "偏心管大小头", "ECC. REDUCER", "偏心异径", "ECC SWAGE", "ECC REDUCER", "EccentricSwagedNippl", "eccentricreduce", "ECC.REDUCER", "ECCSWAGE", "ECCREDUCER", "REDUCER ECC", "EOCENTRIC REDUCER", "BOCENTRIC REDUCER", "BCCENTRIC REDUCER", "ECC. SWAGE", "ECC SWAGE NIPPLE", "SWAGE NIPPLE ECC", "偏心异径管 SWAGE NIPPLE"],
    "RO": ["限流孔板"],
    "RT": ["TEE(TR)", "REDUCING TEE", "异径三通", "RED TEE", "REDTEE", "KEDULIIVG IEE", "TEE REDUCED", "Red Te"],
    "RTP": ["异径三通(纵向剖分成对包装)"],
    "RV": ["Relief Valve", "ReliefValve"],
    "SB": ["BLANK AND SPACER", "blank & spacer", "SPACER RING", "插板&垫环,", "‌PADDLE BLANK & SPACER"],
    "SDV": ["SIAMESE DOUBLE VALVE", "连体式双阀门", "SIAMESEDOUBLEVALVE"],
    "SG": ["视镜"],
    "SN": ["室内消火栓"],
    "ST": ["Strainer", "过滤器"],
    "STRV": ["STEAM TRAP", "蒸汽疏水阀", "STEAMTRAP"],
    "SV": ["SAFE VALVE", "安全阀", "SAFEVALVE"],
    "SWG": ["SWG GASKET", "缠绕式垫片", "SPIRAL-WOUND GASKET", "缠绕垫", "SPIRAL WOUND GASKET", "SW Gk"],
    "T": ["straight tee", "Tee", "TEE(TS)", "等径三通", "TEE", "同径三通", "无缝三通", "有缝三通", "Eq Te", "三通"],
    "TBV": ["三通球阀"],
    "TI": ["仪表温度计"],
    "TP": ["等径三通(纵向剖分成对包装)"],
    "TST": ["T-Type Strainer", "T-型过滤器", "thermostatic trap", "温控疏水阀", "T-TypeStrainer", "thermostatictrap"],
    "WPBV": ["WAFER PNEUMATIC BALL VALVE", "WAFERPNEUMATICBALLVALVE"],
    "XYQ": ["消音器"],
    "YST": ["Y-Type Strainer", "Y-型过滤器", "Y-TypeStrainer", "Y型过滤器"]
}


def prepare_classification_data(
    mapping: Dict[str, List[str]],
    output_dir: str,
    train_ratio: float = 0.8
):
    """
    准备分类模型训练数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标签映射
    labels = sorted(mapping.keys())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # 生成所有样本
    all_samples = []
    for code, descriptions in mapping.items():
        for desc in descriptions:
            sample = {
                "text": desc,
                "label": label2id[code],
                "label_name": code
            }
            all_samples.append(sample)
    
    # 打乱顺序
    random.seed(42)
    random.shuffle(all_samples)
    
    # 分割训练集和验证集
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 保存训练集
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 保存验证集
    val_file = output_dir / "val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 保存标签映射
    label_file = output_dir / "labels.json"
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump({
            "label2id": label2id,
            "id2label": id2label,
            "num_labels": len(labels)
        }, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print(f"数据准备完成！")
    print(f"总样本数: {len(all_samples)}")
    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    print(f"类别数: {len(labels)}")
    print(f"")
    print(f"输出文件:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {label_file}")
    
    return train_samples, val_samples, label2id, id2label


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "type_classifier"
    prepare_classification_data(TYPE_MAPPING, output_dir)
