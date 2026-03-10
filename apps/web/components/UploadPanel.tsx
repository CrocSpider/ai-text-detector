"use client";

import { useRef } from "react";

interface UploadPanelProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

export function UploadPanel({ files, onFilesChange }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  function mergeFiles(nextFiles: FileList | null) {
    if (!nextFiles) return;
    const merged = [...files, ...Array.from(nextFiles)];
    const deduped = Array.from(new Map(merged.map((file) => [`${file.name}-${file.size}`, file])).values());
    onFilesChange(deduped);
  }

  function removeFile(fileName: string) {
    onFilesChange(files.filter((file) => file.name !== fileName));
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Upload files</h2>
        <span>TXT, PDF, DOCX, MD, CSV, JSON, HTML, RTF</span>
      </div>

      <button
        type="button"
        className="upload-dropzone"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          mergeFiles(event.dataTransfer.files);
        }}
      >
        <strong>Drag and drop files here</strong>
        <span>or click to browse</span>
      </button>

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        multiple
        onChange={(event) => mergeFiles(event.target.files)}
        accept=".txt,.md,.csv,.json,.html,.htm,.rtf,.docx,.pdf"
      />

      {files.length > 0 ? (
        <div className="file-list">
          {files.map((file) => (
            <div className="file-chip" key={`${file.name}-${file.size}`}>
              <span>{file.name}</span>
              <button type="button" onClick={() => removeFile(file.name)} aria-label={`Remove ${file.name}`}>
                Remove
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="helper-text">You can upload one file or a small batch. Batch jobs return one result card per document.</p>
      )}
    </section>
  );
}
