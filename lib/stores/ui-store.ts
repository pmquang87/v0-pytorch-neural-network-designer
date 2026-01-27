import { create } from 'zustand';
import type { ModelAnalysis } from '../model-analyzer';

// UI state interface
interface UIState {
    // Dialog states
    showCodeDialog: boolean;
    showHelpDialog: boolean;
    showSaveDialog: boolean;
    showLoadDialog: boolean;
    showCodeInputDialog: boolean;

    // Panel states
    showAnalysisPanel: boolean;
    showValidationPanel: boolean;

    // Code generation state
    generatedCode: string;
    isGenerating: boolean;
    copySuccess: boolean;

    // Code import state
    inputCode: string;
    parseErrors: string[];
    parseWarnings: string[];
    unsupportedModules: string[];

    // Model analysis
    modelAnalysis: ModelAnalysis | null;

    // Validation results
    validationResults: { errors: string[]; warnings: string[] } | null;
    liveValidationResults: { errors: string[]; warnings: string[] } | null;

    // Saved models
    savedModels: Array<{ key: string; name: string; timestamp: number }>;
    modelName: string;

    // Actions
    setShowCodeDialog: (show: boolean) => void;
    setShowHelpDialog: (show: boolean) => void;
    setShowSaveDialog: (show: boolean) => void;
    setShowLoadDialog: (show: boolean) => void;
    setShowCodeInputDialog: (show: boolean) => void;
    setShowAnalysisPanel: (show: boolean) => void;
    setShowValidationPanel: (show: boolean) => void;

    setGeneratedCode: (code: string) => void;
    setIsGenerating: (generating: boolean) => void;
    setCopySuccess: (success: boolean) => void;

    setInputCode: (code: string) => void;
    setParseErrors: (errors: string[]) => void;
    setParseWarnings: (warnings: string[]) => void;
    setUnsupportedModules: (modules: string[]) => void;

    setModelAnalysis: (analysis: ModelAnalysis | null) => void;
    setValidationResults: (results: { errors: string[]; warnings: string[] } | null) => void;
    setLiveValidationResults: (results: { errors: string[]; warnings: string[] } | null) => void;

    setSavedModels: (models: Array<{ key: string; name: string; timestamp: number }>) => void;
    setModelName: (name: string) => void;

    // Bulk actions
    resetCodeImportState: () => void;
    closeAllDialogs: () => void;
}

export const useUIStore = create<UIState>()((set) => ({
    // Initial dialog states
    showCodeDialog: false,
    showHelpDialog: false,
    showSaveDialog: false,
    showLoadDialog: false,
    showCodeInputDialog: false,

    // Initial panel states
    showAnalysisPanel: false,
    showValidationPanel: false,

    // Initial code generation state
    generatedCode: '',
    isGenerating: false,
    copySuccess: false,

    // Initial code import state
    inputCode: '',
    parseErrors: [],
    parseWarnings: [],
    unsupportedModules: [],

    // Initial model analysis
    modelAnalysis: null,

    // Initial validation results
    validationResults: null,
    liveValidationResults: null,

    // Initial saved models
    savedModels: [],
    modelName: '',

    // Dialog actions
    setShowCodeDialog: (show) => set({ showCodeDialog: show }),
    setShowHelpDialog: (show) => set({ showHelpDialog: show }),
    setShowSaveDialog: (show) => set({ showSaveDialog: show }),
    setShowLoadDialog: (show) => set({ showLoadDialog: show }),
    setShowCodeInputDialog: (show) => set({ showCodeInputDialog: show }),
    setShowAnalysisPanel: (show) => set({ showAnalysisPanel: show }),
    setShowValidationPanel: (show) => set({ showValidationPanel: show }),

    // Code generation actions
    setGeneratedCode: (code) => set({ generatedCode: code }),
    setIsGenerating: (generating) => set({ isGenerating: generating }),
    setCopySuccess: (success) => set({ copySuccess: success }),

    // Code import actions
    setInputCode: (code) => set({ inputCode: code }),
    setParseErrors: (errors) => set({ parseErrors: errors }),
    setParseWarnings: (warnings) => set({ parseWarnings: warnings }),
    setUnsupportedModules: (modules) => set({ unsupportedModules: modules }),

    // Analysis actions
    setModelAnalysis: (analysis) => set({ modelAnalysis: analysis }),
    setValidationResults: (results) => set({ validationResults: results }),
    setLiveValidationResults: (results) => set({ liveValidationResults: results }),

    // Saved models actions
    setSavedModels: (models) => set({ savedModels: models }),
    setModelName: (name) => set({ modelName: name }),

    // Bulk actions
    resetCodeImportState: () =>
        set({
            inputCode: '',
            parseErrors: [],
            parseWarnings: [],
            unsupportedModules: [],
        }),

    closeAllDialogs: () =>
        set({
            showCodeDialog: false,
            showHelpDialog: false,
            showSaveDialog: false,
            showLoadDialog: false,
            showCodeInputDialog: false,
        }),
}));

// Selector hooks for common operations
export const useDialogState = () =>
    useUIStore((state) => ({
        showCodeDialog: state.showCodeDialog,
        showHelpDialog: state.showHelpDialog,
        showSaveDialog: state.showSaveDialog,
        showLoadDialog: state.showLoadDialog,
        showCodeInputDialog: state.showCodeInputDialog,
    }));

export const usePanelState = () =>
    useUIStore((state) => ({
        showAnalysisPanel: state.showAnalysisPanel,
        showValidationPanel: state.showValidationPanel,
    }));

export const useValidation = () =>
    useUIStore((state) => ({
        validationResults: state.validationResults,
        liveValidationResults: state.liveValidationResults,
    }));
