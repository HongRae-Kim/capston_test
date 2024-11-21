import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import LoginPage from "./pages/LoginPage";
import RegisterPage from './pages/RegisterPage';
import PrivateRoute from './components/PrivateRoute';
import { AuthProvider, useAuth } from './contexts/AuthContext';

function App() {
  return (
      <AuthProvider> {/* 로그인 상태를 관리하는 AuthProvider로 전체 애플리케이션 감싸기 */}
        <Router>
          <Navbar /> {/* 로그인 상태에 따라 네비게이션 바를 동적으로 표시 */}
          <Routes>
            {/* 로그인 상태 확인 및 리다이렉트 */}
            <Route path="/" element={<RequireAuth />} />

            {/* 공개된 라우트 */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            {/* Private Routes (로그인된 사용자만 접근 가능) */}
            <Route
                path="/home"
                element={
                  <PrivateRoute>
                    <HomePage />
                  </PrivateRoute>
                }
            />
            <Route
                path="/upload"
                element={
                  <PrivateRoute>
                    <UploadPage />
                  </PrivateRoute>
                }
            />
            <Route
                path="/results"
                element={
                  <PrivateRoute>
                    <ResultsPage />
                  </PrivateRoute>
                }
            />
          </Routes>
        </Router>
      </AuthProvider>
  );
}

// 사용자가 로그인하지 않았으면 로그인 페이지로 이동시키는 컴포넌트
function RequireAuth() {
  const { token } = useAuth();

  if (!token) {
    return <Navigate to="/login" />;
  }

  return <Navigate to="/home" />;
}

export default App;
