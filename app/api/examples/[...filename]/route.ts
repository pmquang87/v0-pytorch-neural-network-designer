import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(
  request: Request,
  { params }: { params: { filename: string[] } }
) {
  const filename = params.filename.join('/');

  // Prevent path traversal
  if (filename.includes('..')) {
    return new NextResponse('Invalid filename', { status: 400 });
  }

  const filePath = path.join(process.cwd(), 'lib', 'examples', filename);

  try {
    const fileContents = await fs.readFile(filePath, 'utf8');
    const data = JSON.parse(fileContents);
    return NextResponse.json(data);
  } catch (error) {
    if (error instanceof SyntaxError) {
        return new NextResponse(`JSON syntax error in ${filename}: ${error.message}`, { status: 500 });
    }
    if (error.code === 'ENOENT') {
      return new NextResponse(`File not found: ${filename}`, { status: 404 });
    }
    console.error(`Error processing file ${filename}:`, error);
    return new NextResponse(`Error reading file ${filename}. See server logs for details.`, { status: 500 });
  }
}
