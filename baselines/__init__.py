"""베이스라인 정책 등록표: 키 → (결과 라벨, 시뮬레이터 생성 함수)

새 베이스라인은 policies.py에 구현하고 POLICIES에 한 줄 추가하면 비교 실험, 실험 실행기, 뷰어에서 모두 쓸 수 있음
"""
from functools import partial

from baselines.policies import (FullBufferSimulator, FillBufferSimulator, PartialSimulator, RandomAssignSimulator,
                                NearestRobotSimulator, MinInsertionSimulator, MarginalCostSimulator,
                                MarginalCostReserveSimulator, GreedyRewardSimulator, OrderPlannedSimulator, PARTIAL_K,
                                RandomReserveSimulator, NearestReserveSimulator, MinInsertionReserveSimulator)

BL6_CD_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)   # BL6* 튜닝 격자 (설정마다 검증 시드로 c_d 선택)

# 정책 키 -> (라벨, 시뮬레이터 생성 함수 (layout, **SimulatorBase 인자))
POLICIES = {
    "BL1":   ("BL1_FullBuffer",          FullBufferSimulator),
    "BL1F":  ("BL1F_FillBuffer",         FillBufferSimulator),
    "BL2":   (f"BL2_Partial_k{PARTIAL_K}", PartialSimulator),
    "BL3":   ("BL3_Random",              RandomAssignSimulator),
    "BL4":   ("BL4_Nearest",             NearestRobotSimulator),
    "BL5":   ("BL5_MinInsertion",        MinInsertionSimulator),
    "BL6":   ("BL6_MarginalCost",        MarginalCostSimulator),
    "BL6D0": ("BL6_MarginalCost_cd0",    partial(MarginalCostSimulator, cost_dist=0.0)),   # 거리비용 c_d = 0
    "GR":    ("Greedy_R",                GreedyRewardSimulator),
    "BL8":   ("BL8_OrderPlanned",        OrderPlannedSimulator),
    # 후보가 없으면 공용 FIFO 큐 대신 로봇별 다음 트립 예약 (baselines.policies.ReserveMixin)
    "BL3R":  ("BL3R_Random_reserve",       RandomReserveSimulator),
    "BL4R":  ("BL4R_Nearest_reserve",      NearestReserveSimulator),
    "BL5R":  ("BL5R_MinInsertion_reserve", MinInsertionReserveSimulator),
    "BL6R":  ("BL6R_MarginalCost_reserve", MarginalCostReserveSimulator),
}
# 논문 번호 BL7 = Greedy_R (정책 키 GR), BL6c0 = BL6D0, BL6c1 = BL6
POLICY_ALIASES = {"BL7": "GR", "BL6c0": "BL6D0", "BL6c1": "BL6"}
# 매개변수 키 (resolve_policy)
#   BL6c{c}      : BL6에서 c_d = c (라벨 BL6_MarginalCost_cd{c}), 예: BL6c0.25. BL6* 튜닝 격자 BL6_CD_GRID
#   BL6Rc{c}     : BL6R에서 c_d = c (라벨 BL6R_MarginalCost_reserve_cd{c})
#   {키}@bf      : 주문 내 토트 배정 순서 bottleneck_first (라벨 뒤 _bf), 예: BL6@bf, BL6c0.5@bf, BL8@bf
BF_SUFFIX = "@bf"


def policy_key(key: str) -> str:
    """별칭(BL7, BL6c0, BL6c1)을 POLICIES 키로 (@bf 꼬리표는 유지)"""
    base, bf = (key[:-len(BF_SUFFIX)], BF_SUFFIX) if key.endswith(BF_SUFFIX) else (key, "")
    return POLICY_ALIASES.get(base, base) + bf


def resolve_policy(key: str):
    """정책 키 → (라벨, 시뮬레이터 생성 함수). 등록 키, 별칭, 매개변수 키(모듈 상단 설명) 지원"""
    key = policy_key(key)
    bf = key.endswith(BF_SUFFIX)
    base = key[:-len(BF_SUFFIX)] if bf else key
    if base in POLICIES:
        label, factory = POLICIES[base]
    elif base.startswith("BL6c"):
        try:
            c = float(base[4:])
        except ValueError:
            raise KeyError(key)
        label, factory = f"BL6_MarginalCost_cd{c:g}", partial(MarginalCostSimulator, cost_dist=c)
    elif base.startswith("BL6Rc"):
        try:
            c = float(base[5:])
        except ValueError:
            raise KeyError(key)
        label, factory = f"BL6R_MarginalCost_reserve_cd{c:g}", partial(MarginalCostReserveSimulator, cost_dist=c)
    else:
        raise KeyError(key)
    if bf:
        label, factory = label + "_bf", partial(factory, tote_order="bottleneck_first")
    return label, factory


def policy_arg(key: str) -> str:
    """argparse type: 해석할 수 있는 정책 키인지 확인"""
    import argparse
    try:
        resolve_policy(key)
    except KeyError:
        raise argparse.ArgumentTypeError(f"알 수 없는 정책 키 {key} (등록 키: {', '.join(POLICIES)}, "
                                         f"별칭: {', '.join(POLICY_ALIASES)}, 매개변수: BL6c<c_d>, <키>@bf)")
    return key


def policy_label(key: str) -> str:
    """정책 키의 결과 라벨"""
    return resolve_policy(key)[0]
