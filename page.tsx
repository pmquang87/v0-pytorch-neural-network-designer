      {/* Model Validation Warning Banner */}
      <div className="p-4 bg-amber-50 border-amber-200 border rounded-md mb-4 mx-4">
        <div className="flex items-start">
          <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="font-medium text-amber-800">Model Validation Results</h3>
            <p className="text-amber-700 text-sm mt-1">Model has issues that need attention.</p>
            <div className="mt-2 bg-white bg-opacity-50 p-2 rounded border border-amber-200">
              <p className="font-medium text-sm text-red-600">Errors (1)</p>
              <p className="text-sm">Shape mismatch between input-1 output and linear-1 input</p>
              <div className="mt-2 text-xs text-amber-800">
                <p className="font-medium">Quick Fix:</p>
                <p>Ensure the linear layer's <code className="bg-amber-100 px-1 rounded">in_features</code> parameter matches the number of features from the input layer.</p>
              </div>
            </div>
            <div className="mt-2 flex gap-2">
              <Button variant="outline" size="sm" className="text-xs h-7" onClick={() => window.location.href = '/examples/ShapeMismatchFix'}>
                View Examples
              </Button>
              <Button variant="outline" size="sm" className="text-xs h-7" onClick={() => window.location.href = '/examples/SimpleMLPExample'}>
                Simple MLP Template
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
          <div className="p-4 border-b border-sidebar-border overflow-y-auto">
            <h2 className="font-semibold text-sidebar-foreground mb-3">Layer Library</h2>
            <div className="space-y-4">
              {/* Input Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Input</div>
                <div className="space-y-2">
