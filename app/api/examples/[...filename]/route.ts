import { NextResponse, type NextRequest } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string[] }> }
) {
  const { filename: filenameParts } = await params;
  const filename = filenameParts.join('/');

  // Prevent path traversal: resolve and verify the path stays inside the
  // examples directory instead of relying on a substring blocklist
  const baseDir = path.resolve(process.cwd(), 'lib', 'examples');
  const filePath = path.resolve(baseDir, filename);
  if (filePath !== baseDir && !filePath.startsWith(baseDir + path.sep)) {
    return new NextResponse('Invalid filename', { status: 400 });
  }

  try {
    const fileContents = await fs.readFile(filePath, 'utf8');
    const data = JSON.parse(fileContents);
    return NextResponse.json(data);
  } catch (error: unknown) {
    if (error instanceof SyntaxError) {
        return new NextResponse(`JSON syntax error in ${filename}: ${error.message}`, { status: 500 });
    }
    if (typeof error === 'object' && error !== null && 'code' in error && (error as { code?: string }).code === 'ENOENT') {
      return new NextResponse(`File not found: ${filename}`, { status: 404 });
    }
    console.error(`Error processing file ${filename}:`, error);
    return new NextResponse(`Error reading file ${filename}. See server logs for details.`, { status: 500 });
  }
}
