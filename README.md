Untuk menjalankan file LaTeX lengkap sebagai komponen web, kita perlu menggunakan kombinasi parser LaTeX ke HTML dan Web Components. Berikut implementasi lengkapnya:

### 1. Persiapan Library
```html
<!-- Load library utama -->
<script src="https://cdn.jsdelivr.net/npm/latex.js@0.12.1/dist/latex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/latex.js@0.12.1/dist/html.min.js"></script>
```

### 2. Membuat Komponen untuk File LaTeX Lengkap
```javascript
class LatexDocument extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.loader = this.createLoader();
  }

  createLoader() {
    const loader = document.createElement('div');
    loader.innerHTML = 'Memuat dokumen...';
    loader.style.cssText = `
      padding: 20px;
      font-style: italic;
      color: #666;
    `;
    return loader;
  }

  async connectedCallback() {
    this.shadowRoot.appendChild(this.loader);
    
    try {
      const response = await fetch(this.getAttribute('src'));
      const latexCode = await response.text();
      
      const generator = new LaTeXJS.HtmlGenerator({
        hyphenate: false,
      });
      
      const parser = new LaTeXJS.Parser({
        generator: generator,
      });
      
      parser.parse(latexCode);
      
      const style = document.createElement('style');
      style.textContent = `
        .latex-document {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: serif;
        }
        .latex-document h1, .latex-document h2 {
          color: #2c3e50;
        }
        .latex-document p {
          line-height: 1.6;
        }
      `;
      
      const wrapper = document.createElement('div');
      wrapper.className = 'latex-document';
      wrapper.innerHTML = generator.styles + generator.documentFragment.innerHTML;
      
      this.shadowRoot.replaceChildren(style, wrapper);
      
    } catch (error) {
      this.shadowRoot.innerHTML = `Error: ${error.message}`;
    }
  }
}

customElements.define('latex-document', LatexDocument);
```

### 3. Penggunaan dalam HTML
```html
<latex-document src="document.tex"></latex-document>
```

### Contoh File document.tex
```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}

\begin{document}

\title{Contoh Dokumen}
\author{Penulis}
\date{\today}

\maketitle

\section{Introduction}

Persamaan ikonik Einstein:
\[ E = mc^2 \]

\subsection{Contoh Matematika}
\begin{equation}
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
\end{equation}

\begin{itemize}
\item Item pertama
\item Item kedua
\end{itemize}

\end{document}
```

### Fitur yang Didukung:
1. **Struktur Dokumen Lengkap**:
   ```html
   <latex-document src="full-paper.tex"></latex-document>
   ```

2. **Custom Template**:
   ```javascript
   generator.documentClass = article;
   generator.packages = [amsmath, graphicx];
   ```

3. **Interaktivitas**:
   ```javascript
   wrapper.querySelectorAll('.equation').forEach(eq => {
     eq.addEventListener('click', () => {
       eq.style.backgroundColor = '#f0f0f0';
     });
   });
   ```

### Batasan dan Solusi:
1. **Paket LaTeX Tidak Didukung**:
   - Tambahkan polyfill untuk paket tertentu:
   ```javascript
   LaTeXJS.Packages.add('mypackage', {
     macros: {
       '\\mycommand': function(context) {
         // Implementasi custom command
       }
     }
   });
   ```

2. **Render Kompleks**:
   Gunakan Web Worker untuk proses rendering berat:
   ```javascript
   const worker = new Worker('latex-worker.js');
   worker.postMessage({ latex: content });
   ```

3. **Server-Side Rendering**:
   Untuk dokumen sangat besar, gunakan Node.js untuk pre-render:
   ```bash
   npm install latex.js
   ```
   ```javascript
   const { HtmlGenerator, Parser } = require('latex.js');
   
   function renderLatex(content) {
     const gen = new HtmlGenerator();
     const parser = new Parser(gen);
     parser.parse(content);
     return gen.styles + gen.documentFragment.innerHTML;
   }
   ```

### Contoh Integrasi Lanjutan
```html
<latex-document 
  src="thesis.tex"
  style="--primary-color: #2c3e50; --font-size: 1.1em;"
  responsive
  lazy-load
></latex-document>
```

```javascript
// Tambahkan observer untuk lazy-load
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.loadContent();
    }
  });
});

document.querySelectorAll('latex-document[lazy-load]').forEach(el => {
  observer.observe(el);
});
```

Untuk dokumen LaTeX yang sangat kompleks, pertimbangkan menggunakan solusi hybrid:
1. Gunakan MathJax/KaTeX untuk persamaan matematika
2. Remark untuk konversi Markdown
3. Pandoc untuk konversi format dokumen

```javascript
// Contoh integrasi dengan MathJax
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  startup: {
    typeset: false
  }
};

component.shadowRoot.querySelectorAll('.math').forEach(math => {
  MathJax.typeset([math]);
});
```
