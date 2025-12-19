"use client";

import { useState } from 'react';

export default function Home() {
  const [story, setStory] = useState('A happy robot finds a flower. He feels very surprised. Then he dances with joy.');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const generateComic = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ story_text: story }),
      });

      if (!res.ok) {
        throw new Error(`Error: ${res.statusText}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center py-20 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-4xl space-y-12">
        {/* Header Section */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 pb-2">
            ComicGen AI
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Transform your imagination into professional pencil-art comic strips in seconds.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 p-8 rounded-2xl shadow-2xl transition-all duration-300 hover:shadow-blue-900/10">
          <label className="block text-sm font-medium text-slate-300 mb-3 uppercase tracking-wider">
            Your Story Script
          </label>
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl blur opacity-20 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
            <textarea
              className="relative w-full h-40 p-4 bg-slate-900/90 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-slate-100 placeholder-slate-500 resize-none transition-all duration-200"
              value={story}
              onChange={(e) => setStory(e.target.value)}
              placeholder="Once upon a time in a digital galaxy..."
            />
          </div>
          
          <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-slate-500">
              * Tips: Describe characters and actions clearly for best results.
            </p>
            <button
              onClick={generateComic}
              disabled={loading || !story}
              className={`
                group relative px-8 py-3 rounded-full font-bold text-white transition-all duration-200
                ${loading 
                  ? 'bg-slate-700 cursor-not-allowed opacity-70' 
                  : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 hover:shadow-lg hover:shadow-blue-500/25 active:scale-95'
                }
              `}
            >
              <span className="flex items-center gap-2">
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Creating Magic...
                  </>
                ) : (
                  <>
                    Generate Comic
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </>
                )}
              </span>
            </button>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-200 flex items-center gap-3">
              <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {error}
            </div>
          )}
        </div>

        {/* Result Section */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Your Masterpiece
              </h2>
              <a
                href={`data:image/jpeg;base64,${result.image_base64}`}
                download="comic_strip.jpg"
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-white font-medium transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </a>
            </div>
            
            <div className="bg-white p-3 rotate-1 hover:rotate-0 transition-transform duration-500 shadow-2xl skew-y-1 hover:skew-y-0">
               <div className="border border-slate-200">
                 <img
                   src={`data:image/jpeg;base64,${result.image_base64}`}
                   alt="Generated Comic"
                   className="w-full h-auto block"
                 />
               </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
