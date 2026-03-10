import Link from "next/link";

import { ResultCard } from "@/components/ResultCard";
import { getResult } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ResultDetailsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const result = await getResult(id);

  return (
    <main className="page-shell page-shell--narrow">
      <div className="page-backlink">
        <Link href="/">Back to analyzer</Link>
      </div>
      <ResultCard result={result} showLink={false} />
    </main>
  );
}
