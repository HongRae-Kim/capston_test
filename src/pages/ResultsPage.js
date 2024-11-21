import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import '../css/ResultsPage.css';

// Chart.js 구성 요소 등록
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// 상수 정의
const DEFAULT_DATA = '데이터 없음';

function ResultsPage() {
    // sessionStorage에서 resultData를 가져옵니다.
    const storedResultData = sessionStorage.getItem('resultData');
    const resultData = storedResultData ? JSON.parse(storedResultData) : null;

    // 결과 데이터를 객체로 한 번에 관리하는 방식으로 리팩토링
    const [resultState, setResultState] = useState({
        vascularAge: DEFAULT_DATA,
        vascularScore: DEFAULT_DATA,
        agingSpeed: DEFAULT_DATA,
    });

    useEffect(() => {
        if (resultData) {
            updateResultData(resultData);
        }
    }, [resultData]);

    const updateResultData = (data) => {
        setResultState({
            vascularAge: data.vascular_age ?? DEFAULT_DATA,
            vascularScore: data.vascular_score ?? DEFAULT_DATA,
            agingSpeed: data.aging_speed ?? DEFAULT_DATA,
        });
    };

    return (
        <div className="container results-page">
            <ResultSummary resultState={resultState} />
            <ResultExplanation resultState={resultState} />
            <ChartSection />
        </div>
    );
}

// 결과 요약 섹션 컴포넌트
function ResultSummary({ resultState }) {
    const { vascularAge, vascularScore, agingSpeed } = resultState;
    return (
        <div className="result-summary">
            <h2>혈관 분석 결과</h2>
            <div className="summary-card">
                <h3>당신의 혈관 나이: <span className="vascular-age">{vascularAge}</span></h3>
                <h3>당신의 혈관 점수: <span className="vascular-score">{vascularScore}</span></h3>
                <h3>노화 속도: <span className="aging-speed">{agingSpeed}</span></h3>
            </div>
        </div>
    );
}

// 결과 해석 섹션 컴포넌트
function ResultExplanation({ resultState }) {
    const { vascularAge, vascularScore, agingSpeed } = resultState;

    return (
        <div className="result-explanation">
            <h3>결과 해석</h3>
            <div className="explanation-content">
                <ul>
                    <li className="explanation-item">
                        <strong>혈관 나이:</strong> <span className="highlight">{vascularAge}세</span>는 현재 당신의 생물학적 나이와 혈관 건강 상태를 종합적으로 분석하여 도출된 결과입니다.
                        이 수치는 <span className="highlight">심박 변화</span>와 <span className="highlight">혈관 탄력성</span> 분석을 바탕으로 측정되었습니다.
                    </li>
                    <li className="explanation-item">
                        <strong>혈관 점수:</strong> <span className="highlight">{vascularScore}점</span> (0-100점 기준)으로, 혈관 건강을 평가하는 지표입니다.
                        이 점수는 <span className="highlight">혈관의 유연성</span>, <span className="highlight">혈류 속도</span>, <span className="highlight">심장 기능</span> 등을 종합적으로 평가한 것입니다.
                    </li>
                    <li className="explanation-item">
                        <strong>노화 속도:</strong> <span className="highlight">{agingSpeed !== DEFAULT_DATA && agingSpeed > 1 ? '빠름' : '정상'}</span>은 현재 혈관이 평균적인 노화 속도에 비해 얼마나 빠르게 진행되는지를 나타냅니다.
                        <span className="highlight">스트레스</span>, <span className="highlight">생활 습관</span> 등이 이 지표에 영향을 미칠 수 있습니다.
                    </li>
                </ul>
                <p className="note">각 항목에 대한 설명을 바탕으로 자신의 건강 상태를 보다 명확히 이해하고, 필요한 조치를 취해보세요.</p>
            </div>
        </div>
    );
}

// 차트 섹션 컴포넌트
function ChartSection() {
    return (
        <div className="chart-section">
            <ComparisonBarChart />
        </div>
    );
}

// 연령대 비교를 위한 바 차트 컴포넌트
function ComparisonBarChart() {
    const data = {
        labels: ['나', '동일 연령대 평균', '건강한 연령대 평균'],
        datasets: [
            {
                label: '혈관 나이 비교',
                data: [47, 50, 45], // 예시 데이터
                backgroundColor: ['rgba(54, 162, 235, 0.2)', 'rgba(255, 99, 132, 0.2)', 'rgba(75, 192, 192, 0.2)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)'],
                borderWidth: 1,
            },
        ],
    };

    return (
        <div className="comparison-bar-chart">
            <h3>내 혈관 나이와 동일 연령대 비교</h3>
            <Bar data={data} />
        </div>
    );
}

export default ResultsPage;
