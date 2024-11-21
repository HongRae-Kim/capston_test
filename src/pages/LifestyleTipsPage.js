import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../css/LifestyleTipsPage.css';

function LifestyleTipsPage({ vascularAge, agingSpeed }) {
    const [tips, setTips] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // 서버에서 팁을 가져오는 useEffect 훅
    useEffect(() => {
        const fetchLifestyleTips = async () => {
            try {
                setLoading(true);
                const response = await axios.post('http://localhost:5080/lifestyle_tips', {
                    vascular_age: vascularAge,
                    aging_speed: agingSpeed
                });

                setTips(response.data.lifestyle_tips);
            } catch (err) {
                setError('건강 관리 팁을 가져오는 도중 오류가 발생했습니다.');
            } finally {
                setLoading(false);
            }
        };

        fetchLifestyleTips();
    }, [vascularAge, agingSpeed]);

    return (
        <div className="container tips-page">
            <h2>개인 맞춤 건강 관리 팁</h2>
            {loading ? (
                <p>건강 관리 팁을 로딩 중입니다...</p>
            ) : error ? (
                <div className="alert alert-danger">
                    {error}
                </div>
            ) : (
                <div className="tips-container">
                    <ul>
                        {tips.map((tip, index) => (
                            <li key={index} className="tip-item">
                                {tip}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default LifestyleTipsPage;
