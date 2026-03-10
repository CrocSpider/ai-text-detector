interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
}

export function TextInput({ value, onChange }: TextInputProps) {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Paste text</h2>
        <span>English-first MVP</span>
      </div>
      <textarea
        className="text-area"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Paste a passage, article, memo, essay, or email thread here. The analyzer will preserve paragraph boundaries and return a cautious risk estimate."
      />
    </section>
  );
}
