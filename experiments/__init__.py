"""실험 코드 (저장소 최상위에서 python -m experiments.<모듈>)

파일 이름 규칙
  configs/*.yaml  실험 설정 (이름 = 결과 폴더 runs/<name>의 prefix)
  run_*.py        실험 실행 → runs/ 에 기록
  analysis.py     runs/ 의 저장 결과를 사후 분석 → results/
  check_*.py      점검·부록 도구
  scripts/*.sh    단계별 실행 스크립트
  tests/          단위 테스트와 회귀 테스트
"""
