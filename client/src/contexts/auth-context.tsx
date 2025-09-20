import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User } from '@shared/schema';
import { apiRequest } from '@/lib/queryClient';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, username: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on app start
    const checkAuth = async () => {
      try {
        const savedUser = localStorage.getItem('auth-user');
        const token = localStorage.getItem('auth-token');
        
        if (savedUser && token) {
          const userData = JSON.parse(savedUser);
          // Verify token is still valid by fetching fresh data
          const response = await apiRequest('GET', `/api/users/${userData.id}`);
          if (response.ok) {
            const freshUser = await response.json();
            setUser(freshUser);
          } else {
            // Token expired or invalid, clear auth data
            localStorage.removeItem('auth-user');
            localStorage.removeItem('auth-token');
          }
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        localStorage.removeItem('auth-user');
        localStorage.removeItem('auth-token');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await apiRequest('POST', '/api/auth/login', { email, password });
      
      if (!response.ok) {
        throw new Error('Invalid credentials');
      }
      
      const userData = await response.json();
      
      // Extract token from user data
      const { token, ...userWithoutToken } = userData;
      
      setUser(userWithoutToken);
      localStorage.setItem('auth-user', JSON.stringify(userWithoutToken));
      
      // Store token separately for security
      if (token) {
        localStorage.setItem('auth-token', token);
      }
    } catch (error) {
      throw new Error('Login failed');
    }
  };

  const register = async (name: string, email: string, username: string, password: string) => {
    try {
      const response = await apiRequest('POST', '/api/auth/register', {
        name,
        email,
        username,
        password,
      });
      
      if (!response.ok) {
        throw new Error('Registration failed');
      }
      
      const userData = await response.json();
      setUser(userData);
      localStorage.setItem('auth-user', JSON.stringify(userData));
    } catch (error) {
      throw new Error('Registration failed');
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth-user');
    localStorage.removeItem('auth-token');
  };

  const value = {
    user,
    login,
    register,
    logout,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}