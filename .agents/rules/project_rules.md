---
name: project-rules
description: 물리와 프로그래밍 6조 팀 프로젝트의 코딩 스타일, 데이터 관리 및 구글 드라이브 연동 규칙
---

# Project & Coding Rules

1. **Python 환경**:
   - `.venv` 가상환경 내에서 실행
   - `python-dotenv`를 통한 환경 변수 관리

2. **Google Drive 정책**:
   - 데이터 원본은 구글 드라이브 `6조 팀플/데이터` 폴더(`1O7FLglb-CYOe0rZsfLJEmlngQmv-Pxrl`)에서 관리
   - 대용량 데이터는 Git이 아닌 구글 드라이브를 통해 팀원 간 공유

3. **코드 품질**:
   - 함수 및 클래스 단위의 Type Hint 명시
   - 한국어 주석 및 Docstring을 통해 팀원 간 가독성 보장
   - 테스트 커버리지 유지를 위해 `tests/` 내 테스트 지속적 업데이트
